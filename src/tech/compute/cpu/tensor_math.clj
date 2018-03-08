(ns tech.compute.cpu.tensor-math
  (:require [tech.datatype.core :refer [v-aget v-aset] :as dtype]
            [tech.datatype.base :as dtype-base]
            [tech.datatype.marshal :as marshal]
            [tech.tensor.math :as tm]
            [clojure.math.combinatorics :as combo]
            [tech.compute.cpu.driver
             :refer [datatype->view-cast-fn
                     datatype->cast-fn
                     array-view-iterator]
             :as cpu-driver]
            [tech.compute.driver :as compute-drv]
            [think.parallel.core :as parallel]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.math-util :as cmu]
            [tech.compute.driver :as drv]
            [think.resource.core :as resource]
            [tech.tensor :as ct]
            [tech.tensor.dimensions :as ct-dims]
            [clojure.core.matrix.stats :as stats])
  (:import [tech.compute.cpu.driver CPUStream]
           [com.github.fommil.netlib BLAS]
           [tech.datatype DoubleArrayView FloatArrayView
            LongArrayView IntArrayView ShortArrayView ByteArrayView]
           [java.security SecureRandom]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


;;Need the interface to get correct type hinting to avoid boxing/unboxing every index.
(definterface ElemIdxToAddressFunction
  (^long idx_to_address [^long arg]))


;;This is the only one that will work with indirect addressing.
(defrecord GeneralElemIdxToAddr [rev-shape rev-strides rev-max-shape]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (ct-dims/elem-idx->addr rev-shape rev-strides rev-max-shape arg)))


(defrecord ElemIdxToAddr [^ints rev-shape ^ints rev-strides ^ints rev-max-shape]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (ct-dims/elem-idx->addr-ary rev-shape rev-strides rev-max-shape arg)))


(defrecord SimpleElemIdxToAddr []
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   arg))


(defrecord SimpleBcastAddr [^long elem-count ^long bcast-amt]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (rem (quot arg bcast-amt)
        elem-count)))


(defrecord SimpleRepeatAddr [^long elem-count]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (rem arg elem-count)))

(defn- ensure-simple-tensor
  [tensor]
  (let [dims (:dimensions tensor)
        dense? (ct-dims/dense? dims)
        increasing? (ct-dims/access-increasing? dims)
        direct? (ct-dims/direct? dims)]
    (when-not (and dense? increasing? direct?)
      (throw (ex-info "Tensors used for indexing must be direct, increasing, and dense"
                      {:dense? dense?
                       :increasing? increasing?
                       :direct? direct?
                       :dimensions dims})))
    tensor))

(defn ^:private get-elem-dims->address
  ^ElemIdxToAddressFunction [dims max-shape]
  ;;Special cases here for speed
  (let [dense? (ct-dims/dense? dims)
        increasing? (ct-dims/access-increasing? dims)
        ;;Any indirect addressing?
        direct? (ct-dims/direct? dims)
        min-shape (drop-while #(= 1 %) (ct-dims/shape dims))]
    (cond
      ;;Special case for indexes that increase monotonically
      (and direct?
           (= (:shape dims)
              max-shape)
           dense?
           increasing?)
      (->SimpleElemIdxToAddr)
      ;;Special case for broadcasting a vector across an image (like applying bias).
      (and direct?
           (= (ct-dims/ecount dims)
              (apply max (ct-dims/shape dims)))
           dense?
           increasing?)
      (let [ec (ct-dims/ecount dims)
            ec-idx (long
                    (->> (map-indexed vector (ct-dims/left-pad-ones (ct-dims/shape dims) max-shape))
                         (filter #(= ec (second %)))
                         (ffirst)))
            broadcast-amt (long (apply * 1 (drop (+ 1 ec-idx) max-shape)))]
        (->SimpleBcastAddr ec broadcast-amt))
      (and direct?
           dense?
           increasing?
           (= min-shape
              (take-last (count min-shape) max-shape)))
      (->SimpleRepeatAddr (ct-dims/ecount dims))
      :else
      (let [{:keys [reverse-shape reverse-strides]} (ct-dims/->reverse-data dims max-shape)]
        (if direct?
          (->ElemIdxToAddr (int-array reverse-shape) (int-array reverse-strides)
                           (int-array (vec (reverse max-shape))))
          (do
            #_(clojure.pprint/pprint {:reverse-shape reverse-shape
                                      :reverse-strides reverse-strides
                                      :reverse-max-shape (ct-dims/reversev max-shape)})
            (->GeneralElemIdxToAddr (mapv (fn [item]
                                            (if (number? item)
                                              item
                                              ;;dtype/get-value works on pure buffers.
                                              (ct/tensor->buffer
                                               (ensure-simple-tensor item))))
                                          reverse-shape)
                                    reverse-strides
                                    (ct-dims/reversev max-shape))))))))


(defmacro ^:private assign-constant-impl
  [view-type view-cast-fn _ dtype-cast-fn]
  `(vector
    (dtype/get-datatype (~dtype-cast-fn 0))
    (fn [buffer# dimensions# value# n-elems#]
      (let [n-elems# (long n-elems#)
            buffer# (~view-cast-fn buffer#)
            idx->address# (get-elem-dims->address dimensions# (get dimensions# :shape))
            value# (~dtype-cast-fn value#)]
        (parallel/parallel-for
         idx# n-elems#
         (v-aset buffer# (.idx_to_address idx->address# idx#) value#))))))


(def ^:private assign-constant-map
  (->> (array-view-iterator assign-constant-impl)
       (into {})))

(defmacro ^:private datatype->cast-fn-symbol
  [dtype]
  (condp = dtype
    :int8 `byte
    :int16 `short
    :int32 `int
    :int64 `long
    :float32 `float
    :float64 `double))


(defn- generate-datatype-combinations
  []
  (let [all-dtypes dtype-base/datatypes]
    (for [lhs all-dtypes
          rhs all-dtypes]
      [lhs rhs])))


(defn max-shape-from-dimensions
  [& args]
  (-> (apply ct-dims/dimension-seq->max-shape args)
      :max-shape))


(defmacro ^:private marshalling-assign-fn
  [lhs-dtype rhs-dtype]
  `(fn [dest# dest-dim#
        src# src-dim#
        n-elems#]
     (let [dest# (datatype->view-cast-fn ~lhs-dtype dest#)
           src# (datatype->view-cast-fn ~rhs-dtype src#)
           max-shape# (max-shape-from-dimensions dest-dim# src-dim#)
           dest-idx->address# (get-elem-dims->address dest-dim# max-shape#)
           src-idx->address# (get-elem-dims->address src-dim# max-shape#)
           n-elems# (long n-elems#)]
       (parallel/parallel-for
        idx# n-elems#
        (v-aset dest# (.idx_to_address dest-idx->address# idx#)
                      (datatype->cast-fn
                       ~lhs-dtype
                       (v-aget src# (.idx_to_address src-idx->address# idx#))))))))


(defmacro ^:private generate-all-marshalling-assign-fns
  []
  (mapv (fn [[lhs-dtype rhs-dtype :as combo]]
          [combo `(marshalling-assign-fn ~lhs-dtype ~rhs-dtype)])
        (generate-datatype-combinations)))


(def ^:private assign!-map
  (->> (generate-all-marshalling-assign-fns)
       (into {})))


(defmacro ^:private perform-unary-op-impl
  [operation x]
  (condp = operation
    :floor `(Math/floor (double ~x))
    :ceil `(Math/ceil (double ~x))
    :round `(Math/round (double ~x))
    :- `(- ~x)
    :tanh `(Math/tanh (double ~x))
    :logistic `(/ 1.0
                  (+ 1.0 (Math/exp (- ~x))))
    :exp `(Math/exp (double ~x))
    :sqrt `(Math/sqrt (double ~x))
    :noop `(double ~x)))


(defmacro ^:private unary-accum!-impl
  [datatype operation]
  `(fn [dest# dest-dims# dest-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# (get dest-dims# :shape))
           dest-alpha# (datatype->cast-fn ~datatype dest-alpha#)]
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [dest-idx# (.idx_to_address dest-idx->address# idx#)]
                (v-aset dest# dest-idx#
                              (datatype->cast-fn
                               ~datatype
                               (perform-unary-op-impl ~operation (* (v-aget dest# dest-idx#)
                                                                    dest-alpha#)))))))))


(defmacro ^:private unary-op!-impl
  [datatype operation]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           n-elems# (long n-elems#)]
       (parallel/parallel-for
        idx# n-elems#
        (v-aset dest# (.idx_to_address dest-idx->address# idx#)
                (datatype->cast-fn
                 ~datatype
                 (perform-unary-op-impl ~operation (* (v-aget x# (.idx_to_address x-idx->address# idx#))
                                                      x-alpha#))))))))


(defmacro unary-op-table-impl
  []
  (->> (for [dtype dtype-base/datatypes
             op ct/unary-operations]
         [[dtype op] {:unary-accum! `(unary-accum!-impl ~dtype ~op)
                      :unary-op! `(unary-op!-impl ~dtype ~op)}])
       (into {})))


(def ^:private unary-op-table
  (unary-op-table-impl))


(defmacro ^:private perform-operation-impl
  [operation x y]
  (condp = operation
    :+ `(+ ~x ~y)
    :- `(- ~x ~y)
    :/ `(/ ~x ~y)
    :* `(* ~x ~y)
    ;;Math/max and friends aren't defined for all primitives leading to reflection warnings.
    :max `(if (> ~x ~y) ~x ~y)
    :min `(if (> ~x ~y) ~y ~x)
    :bit-and `(bit-and (unchecked-int ~x) (unchecked-int ~y))
    :bit-xor `(bit-xor (unchecked-int ~x) (unchecked-int ~y))
    :eq `(if (= ~x ~y)
           1
           0)
    :> `(if (> ~x ~y)
           1
           0)
    :>= `(if (>= ~x ~y)
           1
           0)
    :< `(if (< ~x ~y)
           1
           0)
    :<= `(if (<= ~x ~y)
           1
           0)))


(defmacro ^:private perform-op-rev-ops
  [operation reverse-operands? x y]
  (if reverse-operands?
    `(perform-operation-impl ~operation ~y ~x)
    `(perform-operation-impl ~operation ~x ~y)))


(defmacro ^:private binary-accum-constant!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-dims# dest-alpha#
        scalar#
        n-elems#]
     (let [n-elems# (long n-elems#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# (get dest-dims# :shape))
           scalar# (datatype->cast-fn ~datatype scalar#)
           dest-alpha# (datatype->cast-fn ~datatype dest-alpha#)]
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [dest-idx# (.idx_to_address dest-idx->address# idx#)]
                (v-aset dest# dest-idx#
                              (datatype->cast-fn
                               ~datatype
                               (perform-op-rev-ops ~operation ~reverse-operands?
                                                   (* (v-aget dest# dest-idx#) dest-alpha#)
                                                   scalar#))))))))


(defmacro binary-accum-constant-table
  []
  (->> (for [dtype dtype-base/datatypes
             op ct/binary-operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-accum-constant!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-accum-constant-table
  (binary-accum-constant-table))


(defmacro ^:private binary-op-constant!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        scalar#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           scalar# (datatype->cast-fn ~datatype scalar#)]
       (parallel/parallel-for
        idx# (long n-elems#)
        (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
              x-idx# (.idx_to_address x-idx->address# idx#)]
          (v-aset dest# dest-idx#
                        (datatype->cast-fn
                         ~datatype
                         (perform-op-rev-ops ~operation ~reverse-operands?
                                             (* (v-aget x# x-idx#) x-alpha#)
                                             scalar#))))))))


(defmacro binary-op-constant-table
  []
  (->> (for [dtype dtype-base/datatypes
             op ct/binary-operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-op-constant!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-op-constant-table
  (binary-op-constant-table))


(defmacro ^:private binary-accum!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-dims# dest-alpha#
        y# y-dims# y-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# y-dims#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           dest-alpha# (datatype->cast-fn ~datatype dest-alpha#)
           y# (datatype->view-cast-fn ~datatype y#)
           y-idx->address# (get-elem-dims->address y-dims# max-shape#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)]
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
                    y-idx# (.idx_to_address y-idx->address# idx#)]
                (v-aset dest# dest-idx#
                              (datatype->cast-fn
                               ~datatype
                               (perform-op-rev-ops ~operation ~reverse-operands?
                                                   (* (v-aget dest# dest-idx#) dest-alpha#)
                                                   (* (v-aget y# y-idx#) y-alpha#)))))))))


(defmacro binary-accum-table
  []
  (->> (for [dtype dtype-base/datatypes
             op ct/binary-operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-accum!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-accum-table
  (binary-accum-table))



(defmacro ^:private binary-op!-impl
  [datatype operation]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        y# y-dims# y-alpha#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims# y-dims#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y# (datatype->view-cast-fn ~datatype y#)
           y-idx->address# (get-elem-dims->address y-dims# max-shape#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)]
       (parallel/parallel-for
        idx# (long n-elems#)
        (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
              x-idx# (.idx_to_address x-idx->address# idx#)
              y-idx# (.idx_to_address y-idx->address# idx#)]
          (v-aset dest# dest-idx#
                        (datatype->cast-fn
                         ~datatype
                         (perform-operation-impl ~operation
                                                 (* (v-aget x# x-idx#) x-alpha#)
                                                 (* (v-aget y# y-idx#) y-alpha#)))))))))


(defmacro binary-op-table-impl
  []
  (->> (for [dtype dtype-base/datatypes
             op ct/binary-operations]
         [[dtype op] `(binary-op!-impl ~dtype ~op)])
       (into {})))


(def ^:private binary-op-table
  (binary-op-table-impl))

(defmacro select-impl
  [x y z]
  `(if (>= ~x 0) ~z ~y))


(defmacro ^:private ternary-op-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        y# y-dims# y-alpha#
        z# z-dims# z-alpha#
        n-elems#
        op#]
     (let [max-shape# (max-shape-from-dimensions dest-dims# x-dims# y-dims# z-dims#)
           d-addr# (get-elem-dims->address dest-dims# max-shape#)
           x-addr# (get-elem-dims->address x-dims# max-shape#)
           y-addr# (get-elem-dims->address y-dims# max-shape#)
           z-addr# (get-elem-dims->address z-dims# max-shape#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           x# (datatype->view-cast-fn ~datatype x#)
           y# (datatype->view-cast-fn ~datatype y#)
           z# (datatype->view-cast-fn ~datatype z#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)
           z-alpha# (datatype->cast-fn ~datatype z-alpha#)]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (v-aset dest# (.idx_to_address d-addr# idx#)
                  (datatype->cast-fn ~datatype
                                     (select-impl (* x-alpha# (v-aget x# (.idx_to_address x-addr# idx#)))
                                                  (* y-alpha# (v-aget y# (.idx_to_address y-addr# idx#)))
                                                  (* z-alpha# (v-aget z# (.idx_to_address z-addr# idx#)))))))))))


(defn arg-order->indexes
  [arg-order]
  (let [order-map (->> (map-indexed #(vector %2 %1) arg-order)
                       (into {}))]
    (mapv #(get order-map %) [:x :y :z])))


(defmacro ^:private ternary-op-constant-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        y# y-dims# y-alpha#
        constant#
        n-elems#
        op# arg-order#]
     (let [max-shape# (max-shape-from-dimensions dest-dims# x-dims# y-dims#)
           d-addr# (get-elem-dims->address dest-dims# max-shape#)
           x-addr# (get-elem-dims->address x-dims# max-shape#)
           y-addr# (get-elem-dims->address y-dims# max-shape#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           x# (datatype->view-cast-fn ~datatype x#)
           y# (datatype->view-cast-fn ~datatype y#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)
           arg-indexes# (arg-order->indexes arg-order#)
           [x-dims# y-dims# z-dims#] arg-indexes#]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (let [arg-vec# [(* x-alpha# (v-aget x# (.idx_to_address x-addr# idx#)))
                          (* y-alpha# (v-aget y# (.idx_to_address y-addr# idx#)))
                          constant#]]
           (v-aset dest# (.idx_to_address d-addr# idx#)
                   (datatype->cast-fn ~datatype
                                      (select-impl (datatype->cast-fn ~datatype (get arg-vec# x-dims#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# y-dims#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# z-dims#)))))))))))


(defmacro ^:private ternary-op-constant-constant-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        constant-1#
        constant-2#
        n-elems#
        op# arg-order#]
     (let [max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           d-addr# (get-elem-dims->address dest-dims# max-shape#)
           x-addr# (get-elem-dims->address x-dims# max-shape#)
           dest# (datatype->view-cast-fn ~datatype dest#)
           x# (datatype->view-cast-fn ~datatype x#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           arg-indexes# (arg-order->indexes arg-order#)
           [x-dims# y-dims# z-dims#] arg-indexes#]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (let [arg-vec# [(* x-alpha# (v-aget x# (.idx_to_address x-addr# idx#)))
                          constant-1#
                          constant-2#]]
           (v-aset dest# (.idx_to_address d-addr# idx#)
                   (datatype->cast-fn ~datatype
                                      (select-impl (datatype->cast-fn ~datatype (get arg-vec# x-dims#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# y-dims#))
                                                   (datatype->cast-fn ~datatype (get arg-vec# z-dims#)))))))))))


(defmacro ternary-op-iter
  []
  (->> (for [dtype dtype-base/datatypes]
         [dtype {:ternary-op! `(ternary-op-impl ~dtype)
                 :ternary-op-constant! `(ternary-op-constant-impl ~dtype)
                 :ternary-op-constant-constant! `(ternary-op-constant-constant-impl ~dtype)}])
       (into {})))


(def ^:private ternary-op-table
  (ternary-op-iter))

(defmacro square-expr
  [expr]
  `(let [item# ~expr]
     (* item# item#)))


(defmacro do-unary-reduce-op
  [datatype op input addr in-alpha idx-start idx-stop]
  (condp = op
    :min `(loop [min-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr ~idx-start)))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (min min-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#))))
                     (inc idx#))
              min-val#))
    :max `(loop [max-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr ~idx-start)))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (max max-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#))))
                     (inc idx#))
              max-val#))
    :sum `(loop [sum-val# (* ~in-alpha (v-aget ~input
                                               (.idx_to_address ~addr ~idx-start)))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (+ sum-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#))))
                     (inc idx#))
              sum-val#))
    :mean `(loop [sum-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr ~idx-start)))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (+ sum-val# (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#))))
                     (inc idx#))
              (/ sum-val#
                 (- ~idx-stop ~idx-start))))
    :magnitude `(loop [sum-val# (square-expr (* ~in-alpha (v-aget ~input (.idx_to_address ~addr ~idx-start))))
                       idx# (+ ~idx-start 1)]
                  (if (< idx# ~idx-stop)
                    (recur (+ sum-val# (square-expr (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#)))))
                           (inc idx#))
                    (Math/sqrt sum-val#)))
    :magnitude-squared `(loop [sum-val# (square-expr (* ~in-alpha (v-aget ~input (.idx_to_address ~addr ~idx-start))))
                               idx# (+ ~idx-start 1)]
                          (if (< idx# ~idx-stop)
                            (recur (+ sum-val# (square-expr
                                                (* ~in-alpha (v-aget ~input (.idx_to_address ~addr idx#)))))
                                   (inc idx#))
                            sum-val#))))


(defmacro unary-reduce-impl
  [datatype op]
  `(fn [output# output-dims# input-alpha# input# input-dims#]
     (let [input-shape# (ct-dims/shape input-dims#)
           output-addr# (get-elem-dims->address output-dims# (ct-dims/shape output-dims#))
           input-addr# (get-elem-dims->address input-dims# (ct-dims/shape input-dims#))
           input# (datatype->view-cast-fn ~datatype input#)
           output# (datatype->view-cast-fn ~datatype output#)
           input-alpha# (datatype->cast-fn ~datatype input-alpha#)
           parallelism# (ct-dims/ecount output-dims#)
           iter-amount# (quot (ct-dims/ecount input-dims#)
                              parallelism#)]
       (parallel/parallel-for
        par-idx# parallelism#
        (let [iter-start# (* par-idx# iter-amount#)
              iter-stop# (+ iter-start# iter-amount#)]
         (v-aset output# (.idx_to_address output-addr# par-idx#)
                 (datatype->cast-fn ~datatype
                                    (do-unary-reduce-op ~datatype ~op input# input-addr# input-alpha#
                                                        iter-start# iter-stop#))))))))


(defmacro unary-reduce-iter
  []
  (->> (for [dtype dtype-base/datatypes
             reduce-op ct/unary-reduction-operations]
         [[dtype reduce-op] {:unary-reduce! `(unary-reduce-impl ~dtype ~reduce-op)}])
       (into {})))


(def ^:private unary-reduce-table
  (unary-reduce-iter))


(defmacro ^:private blas-macro-iter
  [inner-macro]
  `{:float64 (~inner-macro marshal/as-double-array-view double .dgemm .dgemv)
    :float32 (~inner-macro marshal/as-float-array-view float .sgemm .sgemv)})


(defmacro ^:private blas-impl
  [cast-fn scalar-cast-fn gemm-op gemv-op]
  `{:gemm (fn [trans-a?# trans-b?# a-row-count# a-col-count# b-col-count#
               ;;Rowstride because blas is row-major (the tensor system is column-major)
               alpha# A# a-rowstride#
               B# b-rowstride#
               beta# C# c-rowstride#]
            (let [trans-a?# (cmu/bool->blas-trans trans-a?#)
                  trans-b?# (cmu/bool->blas-trans trans-b?#)
                  M# (long a-row-count#)
                  N# (long b-col-count#)
                  K# (long a-col-count#)
                  alpha# (~scalar-cast-fn alpha#)
                  beta# (~scalar-cast-fn beta#)
                  A# (~cast-fn A#)
                  B# (~cast-fn B#)
                  C# (~cast-fn C#)
                  A-offset# (.offset A#)
                  B-offset# (.offset B#)
                  C-offset# (.offset C#)
                  A# (.data A#)
                  B# (.data B#)
                  C# (.data C#)]
              (~gemm-op (BLAS/getInstance) trans-a?# trans-b?#
               M# N# K#
               alpha# A# A-offset# a-rowstride#
               B# B-offset# b-rowstride#
               beta# C# C-offset# c-rowstride#)))
    :gemv (fn [trans-a?# a-row-count# a-col-count#
               alpha# A# a-rowstride#
               x# inc-x#
               beta# y# inc-y#]
            (let [a-rowstride# (long a-rowstride#)
                  a-row-count# (long a-row-count#)
                  a-col-count# (long a-col-count#)
                  A# (~cast-fn A#)
                  x# (~cast-fn x#)
                  y# (~cast-fn y#)
                  A-offset# (.offset A#)
                  x-offset# (.offset x#)
                  y-offset# (.offset y#)
                  A# (.data A#)
                  x# (.data x#)
                  y# (.data y#)
                  alpha# (~scalar-cast-fn alpha#)
                  inc-x# (long inc-x#)
                  beta# (~scalar-cast-fn beta#)
                  inc-y# (long inc-y#)]
              (~gemv-op (BLAS/getInstance)
               (cmu/bool->blas-trans trans-a?#)
               a-row-count# a-col-count#
               alpha# A# A-offset# a-rowstride#
               x# x-offset# inc-x#
               beta# y# y-offset# inc-y#)))})


(def ^:private blas-fn-map
  (blas-macro-iter blas-impl))


(defmacro sum-double-var
  "macro to sum a double accumulator.  Note that we are careful
  to avoid adding the first calculated answer to 0.0 as if that answer is very small
  we would introduce roundoff error immediately.  So we need a slightly more complex loop
  in order to avoid adding a small number to 0."
  [idx-var num-iters stmt]
  `(double
    (if (= 0 ~num-iters)
      0.0
      (loop [sum-var# (let [~idx-var 0] ~stmt)
             ~idx-var 1]
        (if (< ~idx-var ~num-iters)
          (recur (+ sum-var# ~stmt) (inc ~idx-var))
          sum-var#)))))


(defonce crap-atom (atom nil))


(defn slice-batches
  [& args]
  (let [num-batches (first (ct/shape (first args)))]
    (map (fn [batch-idx]
           (mapv (fn [arg]
                   (let [dim-count (count (ct/shape arg))]
                     (apply ct/select arg batch-idx (repeat (- dim-count 1) :all))))
                 args))
         (range num-batches))))


(defn ->buffer
  [tensor] (ct/tensor->buffer tensor))


(defn ->dimensions
  [tensor] (ct/tensor->dimensions tensor))


(extend-type CPUStream
  tm/TensorMath
  (assign-constant! [stream tensor value]
    (cpu-driver/with-stream-dispatch stream
      ((get assign-constant-map (dtype/get-datatype tensor))
       (->buffer tensor) (->dimensions tensor) value (ct/ecount tensor))))

  (assign! [stream dest src]
    (cpu-driver/with-stream-dispatch stream
      ((get assign!-map [(dtype/get-datatype dest) (dtype/get-datatype src)])
       (->buffer dest) (->dimensions dest)
       (->buffer src) (->dimensions src)
       (max (ct/ecount src) (ct/ecount dest)))))

  (unary-accum! [stream dest alpha op]
    (cpu-driver/with-stream-dispatch stream
      ((get-in unary-op-table [[(dtype/get-datatype dest) op] :unary-accum!])
       (->buffer dest) (->dimensions dest) alpha (ct/ecount dest))))

  (unary-op! [stream dest x alpha op]
    (cpu-driver/with-stream-dispatch stream
      ((get-in unary-op-table [[(dtype/get-datatype dest) op] :unary-op!])
       (->buffer dest) (->dimensions dest) (->buffer x) (->dimensions x) alpha
       (max (ct/ecount dest) (ct/ecount x)))))

  (binary-accum-constant! [stream dest dest-alpha scalar operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get binary-accum-constant-table [(dtype/get-datatype dest) operation
                                           reverse-operands?])
       (->buffer dest) (->dimensions dest) dest-alpha
       scalar (ct/ecount dest))))

  (binary-op-constant! [stream dest x x-alpha scalar operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get binary-op-constant-table [(dtype/get-datatype dest) operation reverse-operands?])
       (->buffer dest) (->dimensions dest)
       (->buffer x) (->dimensions x) x-alpha
       scalar (max (ct/ecount dest) (ct/ecount x)))))

  (binary-accum! [stream dest dest-alpha y y-alpha operation reverse-operands? dest-requires-cas?]
    (let [n-elems (max (ct/ecount dest) (ct/ecount y))]
      (if dest-requires-cas?
        (cpu-driver/with-stream-dispatch stream
          ((get binary-accum-table [(dtype/get-datatype dest) operation reverse-operands?])
           (->buffer dest) (->dimensions dest) dest-alpha
           (->buffer y) (->dimensions y) y-alpha
           n-elems))
        ;;If the operation does not require a CAS op then we can use the full parallelism of the
        ;;binary op.  Unfortunately if it does then we have to do a lot of things in single-threaded mode.
        (if reverse-operands?
          (tm/binary-op! stream dest y y-alpha dest dest-alpha operation)
          (tm/binary-op! stream dest dest dest-alpha y y-alpha operation)))))

  (binary-op! [stream dest x x-alpha y y-alpha operation]
    (cpu-driver/with-stream-dispatch stream
      ((get binary-op-table [(dtype/get-datatype dest) operation])
       (->buffer dest) (->dimensions dest)
       (->buffer x) (->dimensions x) x-alpha
       (->buffer y) (->dimensions y) y-alpha
       (max (ct/ecount x) (ct/ecount y) (ct/ecount dest)))))

  (ternary-op! [stream dest x x-alpha y y-alpha z z-alpha operation]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op!])
       (->buffer dest) (->dimensions dest)
       (->buffer x) (->dimensions x) x-alpha
       (->buffer y) (->dimensions y) y-alpha
       (->buffer z) (->dimensions z) z-alpha
       (max (ct/ecount x) (ct/ecount y) (ct/ecount z) (ct/ecount dest))
       operation)))

  (ternary-op-constant! [stream dest a a-alpha b b-alpha constant operation arg-order]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op-constant!])
       (->buffer dest) (->dimensions dest)
       (->buffer a) (->dimensions a) a-alpha
       (->buffer b) (->dimensions b) b-alpha
       constant
       (max (ct/ecount a) (ct/ecount b) (ct/ecount dest))
       operation arg-order)))

  (ternary-op-constant-constant! [stream dest a a-alpha const-1 const-2 operation arg-order]
    (cpu-driver/with-stream-dispatch stream
      ((get-in ternary-op-table [(dtype/get-datatype dest) :ternary-op-constant-constant!])
       (->buffer dest) (->dimensions dest)
       (->buffer a) (->dimensions a) a-alpha
       const-1
       const-2
       (max (ct/ecount a) (ct/ecount dest))
       operation arg-order)))

  (unary-reduce! [stream output input-alpha input op]
    (cpu-driver/with-stream-dispatch stream
     ((get-in unary-reduce-table [[(dtype/get-datatype output) op] :unary-reduce!])
      (->buffer output) (->dimensions output)
      input-alpha (->buffer input) (->dimensions input))))

  (gemm! [stream
          C c-colstride
          trans-a? trans-b? alpha
          A a-row-count a-col-count a-colstride
          B b-col-count b-colstride
          beta]
    (cpu-driver/with-stream-dispatch stream
      (cmu/col->row-gemm (get-in blas-fn-map [(dtype/get-datatype C) :gemm])
                         trans-a? trans-b? a-row-count a-col-count b-col-count
                         alpha A a-colstride
                         B b-colstride
                         beta C c-colstride)))

  (gemv! [stream
          c inc-c
          trans-a? alpha
          A a-row-count a-col-count a-colstride
          x inc-x
          beta]
    (cpu-driver/with-stream-dispatch stream
      (cmu/col->row-gemv (get-in blas-fn-map [(dtype/get-datatype c) :gemv])
                         trans-a? a-row-count a-col-count alpha
                         A a-colstride x inc-x beta c inc-c)))


  (rand! [stream dest {:keys [type] :as distribution}]
    (let [rand-view (datatype->view-cast-fn :float32 (->buffer dest))
          elem-count (ct-dims/ecount (->dimensions dest))
          rand-gen (SecureRandom.)]
      (cond
        (= (:type distribution) :gaussian)
        (let [mean (float (:mean distribution))
              multiplier (Math/sqrt (float (:variance distribution)))]
          (c-for [idx 0 (< idx elem-count) (inc idx)]
                 (let [next-rand (+ (* multiplier (.nextGaussian rand-gen))
                                    mean)]
                   (v-aset rand-view idx next-rand))))
        (= (:type distribution) :flat)
        (let [minimum (float (:minimum distribution))
              maximum (float (:maximum distribution))
              range (- maximum minimum)]
         (c-for [idx 0 (< idx elem-count) (inc idx)]
                (v-aset rand-view idx (+ minimum
                                         (* (.nextFloat rand-gen)
                                            range)))))
        :else
        (throw (Exception. (str "Unrecognized distribution: " distribution)))))))


(defn as-tensor
  [java-array]
  (ct/construct-tensor (drv/get-device ct/*stream*)
                       (ct-dims/dimensions [(ct/ecount java-array)])
                       (dtype/->view java-array)))

(defn as-java-array
  [cpu-tensor]
  (drv/sync-with-host ct/*stream*)
  (let [dev-buffer (ct/tensor->buffer cpu-tensor)]
    (condp = (dtype/get-datatype dev-buffer)
      :int8 (.data ^ByteArrayView dev-buffer)
      :int16 (.data ^ShortArrayView dev-buffer)
      :int32 (.data ^IntArrayView dev-buffer)
      :int64 (.data ^LongArrayView dev-buffer)
      :float32 (.data ^FloatArrayView dev-buffer)
      :float64 (.data ^DoubleArrayView dev-buffer)
      )))


(defmacro tensor-context
  [& body]
  `(resource/with-resource-context
     (let [device# (drv/default-device (cpu-driver/driver))
           stream# (drv/create-stream :device device#)]
       (ct/with-stream stream#
         ~@body))))
