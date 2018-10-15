(ns tech.compute.cpu.tensor-math
  (:require [tech.datatype.core :as dtype]
            [tech.datatype.java-primitive :as primitive]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.compute.tensor.math :as tm]
            [tech.compute.driver :as compute-drv]
            [tech.parallel :as parallel]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.math-util :as cmu]
            [tech.compute.driver :as drv]
            [tech.resource :as resource]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.dimensions :as ct-dims]
            [clojure.core.matrix.stats :as stats]
            [clojure.core.matrix :as m]
            [tech.compute.cpu.driver :as cpu-driver])
  (:import [tech.compute.cpu.driver CPUStream]
           [com.github.fommil.netlib BLAS]
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
      (let [{:keys [reverse-shape reverse-strides]}
            (ct-dims/->reverse-data dims max-shape)]
        (if direct?
          (->ElemIdxToAddr (int-array reverse-shape) (int-array reverse-strides)
                           (int-array (vec (reverse max-shape))))
          (do
            #_(clojure.pprint/pprint {:reverse-shape reverse-shape
                                      :reverse-strides reverse-strides
                                      :reverse-max-shape (ct-dims/reversev max-shape)})
            (->GeneralElemIdxToAddr (mapv (fn [item]
                                            (cond
                                              (number? item)
                                              item
                                              (ct/tensor? item)
                                              (ct/tensor->buffer
                                               (ensure-simple-tensor item))
                                              (sequential? item)
                                              (vec item)))
                                          reverse-shape)
                                    reverse-strides
                                    (ct-dims/reversev max-shape))))))))


(defmacro item->typed-nio-buffer
  [datatype item]
  (let [jvm-dtype (unsigned/datatype->jvm-datatype datatype)]
    `(primitive/datatype->buffer-cast-fn
      ~jvm-dtype
      (primitive/->buffer-backing-store ~item))))


(defmacro b-put
  [buffer idx value]
  `(.put ~buffer
         (+ ~idx (.position ~buffer))
         ~value))


(defmacro b-get
  [buffer idx]
  `(.get ~buffer (+ ~idx (.position ~buffer))))


(defmacro ^:private assign-constant-impl
  [datatype]
  `(fn [buffer# dimensions# value# n-elems#]
     (let [n-elems# (long n-elems#)
           buffer# (item->typed-nio-buffer ~datatype buffer#)
           idx->address# (get-elem-dims->address dimensions# (get dimensions# :shape))
           value# (unsigned/datatype->jvm-cast-fn :ignored ~datatype value#)]
       (parallel/parallel-for
        idx# n-elems#
        (b-put buffer#
              (.idx_to_address idx->address# idx#)
              value#)))))


(def all-datatypes (concat primitive/datatypes unsigned/unsigned-datatypes))


(defmacro datatype-iterator
  [iter-macro]
  `(vector
    ~@(for [dtype all-datatypes]
        `(vector ~dtype (~iter-macro ~dtype)))))


(def ^:private assign-constant-map
  (->> (datatype-iterator assign-constant-impl)
       (into {})))


(defn max-shape-from-dimensions
  [& args]
  (-> (apply ct-dims/dimension-seq->max-shape args)
      :max-shape))


(defmacro ^:private marshalling-assign-fn
  [lhs-dtype rhs-dtype]
  (let [rhs-jvm-dtype (unsigned/datatype->jvm-datatype rhs-dtype)
        jvm-dtype (unsigned/datatype->jvm-datatype lhs-dtype)]
    `(fn [dest# dest-dim#
          src# src-dim#
          n-elems#]
       (let [dest# (item->typed-nio-buffer ~lhs-dtype dest#)
             src# (item->typed-nio-buffer ~rhs-dtype src#)
             max-shape# (max-shape-from-dimensions dest-dim# src-dim#)
             dest-idx->address# (get-elem-dims->address dest-dim# max-shape#)
             src-idx->address# (get-elem-dims->address src-dim# max-shape#)
             n-elems# (long n-elems#)]
         (parallel/parallel-for
          idx# n-elems#
          (b-put dest# (.idx_to_address dest-idx->address# idx#)
                (primitive/datatype->unchecked-cast-fn
                 ~rhs-dtype
                 ~jvm-dtype
                 (unsigned/datatype->unchecked-cast-fn
                  ~rhs-jvm-dtype
                  ~rhs-dtype
                  (b-get src# (.idx_to_address src-idx->address# idx#))))))))))


(defmacro ^:private generate-all-marshalling-assign-fns
  []
  (->> unsigned/all-possible-datatype-pairs
       (map (fn [[lhs-dtype rhs-dtype]]
              [[lhs-dtype rhs-dtype]
               `(marshalling-assign-fn ~lhs-dtype ~rhs-dtype)]))
       (into {})))


(def ^:private assign!-map
  (generate-all-marshalling-assign-fns))


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
  (let [src-jvm-dtype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims# dest-alpha#
          n-elems#]
       (let [n-elems# (long n-elems#)
             dest# (item->typed-nio-buffer ~datatype dest#)
             dest-idx->address# (get-elem-dims->address dest-dims#
                                                        (get dest-dims# :shape))
             dest-alpha# (unsigned/datatype->cast-fn :ignored ~datatype dest-alpha#)]
         (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
                (let [dest-idx# (.idx_to_address dest-idx->address# idx#)]
                  (b-put dest# dest-idx#
                        (primitive/datatype->unchecked-cast-fn
                         :float64
                         ~src-jvm-dtype
                         (perform-unary-op-impl
                          ~operation
                          (*
                           (unsigned/datatype->unchecked-cast-fn
                            ~src-jvm-dtype
                            ~datatype
                            (b-get dest# dest-idx#))
                           dest-alpha#))))))))))


(defmacro ^:private unary-op!-impl
  [datatype operation]
  (let [src-jvm-dtype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims#
          x# x-dims# x-alpha#
          n-elems#]
       (let [n-elems# (long n-elems#)
             max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
             dest# (item->typed-nio-buffer ~datatype dest#)
             dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
             x# (item->typed-nio-buffer ~datatype x#)
             x-idx->address# (get-elem-dims->address x-dims# max-shape#)
             x-alpha# (unsigned/datatype->cast-fn :ignored ~datatype x-alpha#)
             n-elems# (long n-elems#)]
         (parallel/parallel-for
          idx# n-elems#
          (b-put dest# (.idx_to_address dest-idx->address# idx#)
                (primitive/datatype->unchecked-cast-fn
                 :float64
                 ~src-jvm-dtype
                 (perform-unary-op-impl
                  ~operation
                  (* (unsigned/datatype->unchecked-cast-fn
                      ~src-jvm-dtype
                      ~datatype
                      (b-get x# (.idx_to_address x-idx->address# idx#)))
                     x-alpha#)))))))))


(defmacro unary-op-table-impl
  []
  (->> (for [dtype all-datatypes
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
    ;;Math/max and friends aren't defined for all primitives leading to reflection
    ;;warnings.
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
  (let [src-jvm-dtype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims# dest-alpha#
          scalar#
          n-elems#]
       (let [n-elems# (long n-elems#)
             dest# (item->typed-nio-buffer ~datatype dest#)
             dest-idx->address# (get-elem-dims->address dest-dims#
                                                        (get dest-dims# :shape))
             scalar# (unsigned/datatype->cast-fn :ignored ~datatype scalar#)
             dest-alpha# (unsigned/datatype->cast-fn :ignored ~datatype dest-alpha#)]
         (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
                (let [dest-idx# (.idx_to_address dest-idx->address# idx#)]
                  (b-put dest# dest-idx#
                        (primitive/datatype->unchecked-cast-fn
                         ~datatype
                         ~src-jvm-dtype
                         (perform-op-rev-ops ~operation ~reverse-operands?
                                             (* (unsigned/datatype->unchecked-cast-fn
                                                 ~src-jvm-dtype
                                                 ~datatype
                                                 (b-get dest# dest-idx#)) dest-alpha#)
                                             scalar#)))))))))


(defmacro binary-accum-constant-table
  []
  (->> (for [dtype all-datatypes
             op ct/binary-operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-accum-constant!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def ^:private binary-accum-constant-table
  (binary-accum-constant-table))


(defmacro datatype->cast-fn
  [dtype val]
  `(unsigned/datatype->cast-fn :ignored ~dtype ~val))


(defmacro store-datatype-cast-fn
  [dtype val]
  (let [jvm-dtype (unsigned/datatype->jvm-datatype dtype)]
    `(unsigned/datatype->unchecked-jvm-cast-fn
      ~dtype
      ~jvm-dtype
      ~val)))


(defmacro read-datatype-cast-fn
  [dtype val]
  (let [jvm-dtype (unsigned/datatype->jvm-datatype dtype)]
    `(unsigned/datatype->unchecked-cast-fn
      ~jvm-dtype
      ~dtype
      ~val)))


(defmacro ^:private binary-op-constant!-impl
  [datatype operation reverse-operands?]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        scalar#
        n-elems#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           dest# (item->typed-nio-buffer ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (item->typed-nio-buffer ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           scalar# (datatype->cast-fn ~datatype scalar#)]
       (parallel/parallel-for
        idx# (long n-elems#)
        (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
              x-idx# (.idx_to_address x-idx->address# idx#)]
          (b-put dest# dest-idx#
                        (store-datatype-cast-fn
                         ~datatype
                         (perform-op-rev-ops ~operation ~reverse-operands?
                                             (* (read-datatype-cast-fn
                                                 ~datatype
                                                 (b-get x# x-idx#)) x-alpha#)
                                             scalar#))))))))


(defmacro binary-op-constant-table
  []
  (->> (for [dtype all-datatypes
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
           dest# (item->typed-nio-buffer ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           dest-alpha# (datatype->cast-fn ~datatype dest-alpha#)
           y# (item->typed-nio-buffer ~datatype y#)
           y-idx->address# (get-elem-dims->address y-dims# max-shape#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)]
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
                    y-idx# (.idx_to_address y-idx->address# idx#)]
                (b-put dest# dest-idx#
                              (store-datatype-cast-fn
                               ~datatype
                               (perform-op-rev-ops ~operation ~reverse-operands?
                                                   (* (read-datatype-cast-fn
                                                       ~datatype
                                                       (b-get dest# dest-idx#))
                                                      dest-alpha#)
                                                   (* (read-datatype-cast-fn
                                                       ~datatype
                                                       (b-get y# y-idx#))
                                                      y-alpha#)))))))))


(defmacro binary-accum-table
  []
  (->> (for [dtype all-datatypes
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
           dest# (item->typed-nio-buffer ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (item->typed-nio-buffer ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y# (item->typed-nio-buffer ~datatype y#)
           y-idx->address# (get-elem-dims->address y-dims# max-shape#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)]
       (parallel/parallel-for
        idx# (long n-elems#)
        (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
              x-idx# (.idx_to_address x-idx->address# idx#)
              y-idx# (.idx_to_address y-idx->address# idx#)]
          (b-put dest# dest-idx#
                        (store-datatype-cast-fn
                         ~datatype
                         (perform-operation-impl ~operation
                                                 (* (read-datatype-cast-fn
                                                     ~datatype
                                                     (b-get x# x-idx#)) x-alpha#)
                                                 (* (read-datatype-cast-fn
                                                     ~datatype
                                                     (b-get y# y-idx#)) y-alpha#)))))))))


(defmacro binary-op-table-impl
  []
  (->> (for [dtype all-datatypes
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
           dest# (item->typed-nio-buffer ~datatype dest#)
           x# (item->typed-nio-buffer ~datatype x#)
           y# (item->typed-nio-buffer ~datatype y#)
           z# (item->typed-nio-buffer ~datatype z#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)
           z-alpha# (datatype->cast-fn ~datatype z-alpha#)]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (b-put dest# (.idx_to_address d-addr# idx#)
                (store-datatype-cast-fn
                 ~datatype
                 (select-impl (* x-alpha# (read-datatype-cast-fn
                                           ~datatype
                                           (b-get x# (.idx_to_address x-addr# idx#))))
                              (* y-alpha# (read-datatype-cast-fn
                                           ~datatype
                                           (b-get y# (.idx_to_address y-addr# idx#))))
                              (* z-alpha# (read-datatype-cast-fn
                                           ~datatype
                                           (b-get z# (.idx_to_address z-addr# idx#))))
                              ))))))))


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
           dest# (item->typed-nio-buffer ~datatype dest#)
           x# (item->typed-nio-buffer ~datatype x#)
           y# (item->typed-nio-buffer ~datatype y#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)
           arg-indexes# (arg-order->indexes arg-order#)
           [x-dims# y-dims# z-dims#] arg-indexes#]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (let [arg-vec# [(* x-alpha# (b-get x# (.idx_to_address x-addr# idx#)))
                          (* y-alpha# (b-get y# (.idx_to_address y-addr# idx#)))
                          constant#]]
           (b-put dest# (.idx_to_address d-addr# idx#)
                 (store-datatype-cast-fn
                  ~datatype
                  (select-impl (unsigned/datatype->unchecked-cast-fn
                                :ingored ~datatype
                                (get arg-vec# x-dims#))
                               (unsigned/datatype->unchecked-cast-fn
                                :ignored ~datatype
                                (get arg-vec# y-dims#))
                               (unsigned/datatype->unchecked-cast-fn
                                :ignored ~datatype
                                (get arg-vec# z-dims#)))))))))))


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
           dest# (item->typed-nio-buffer ~datatype dest#)
           x# (item->typed-nio-buffer ~datatype x#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           arg-indexes# (arg-order->indexes arg-order#)
           [x-dims# y-dims# z-dims#] arg-indexes#]
       (condp = op#
         :select
         (parallel/parallel-for
          idx# n-elems#
          (let [arg-vec# [(* x-alpha# (b-get x# (.idx_to_address x-addr# idx#)))
                          constant-1#
                          constant-2#]]
           (b-put dest# (.idx_to_address d-addr# idx#)
                 (store-datatype-cast-fn
                  ~datatype
                  (select-impl (unsigned/datatype->unchecked-cast-fn
                                :ingored ~datatype
                                (get arg-vec# x-dims#))
                               (unsigned/datatype->unchecked-cast-fn
                                :ignored ~datatype
                                (get arg-vec# y-dims#))
                               (unsigned/datatype->unchecked-cast-fn
                                :ignored ~datatype
                                (get arg-vec# z-dims#))
                               )))))))))


(defmacro ternary-op-iter
  []
  (->> (for [dtype all-datatypes]
         [dtype {:ternary-op! `(ternary-op-impl ~dtype)
                 :ternary-op-constant! `(ternary-op-constant-impl ~dtype)
                 :ternary-op-constant-constant! `(ternary-op-constant-constant-impl
                                                  ~dtype)
                 }])
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
    :min `(loop [min-val# (* ~in-alpha (read-datatype-cast-fn
                                        ~datatype
                                        (b-get ~input
                                              (.idx_to_address ~addr ~idx-start))))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (min min-val# (* ~in-alpha (read-datatype-cast-fn
                                                 ~datatype
                                                 (b-get ~input
                                                       (.idx_to_address ~addr idx#)))))
                     (inc idx#))
              min-val#))
    :max `(loop [max-val# (* ~in-alpha (read-datatype-cast-fn
                                        ~datatype
                                        (b-get ~input (.idx_to_address ~addr
                                                                      ~idx-start))))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (max max-val# (* ~in-alpha (read-datatype-cast-fn
                                                 ~datatype
                                                 (b-get ~input
                                                       (.idx_to_address ~addr idx#)))))
                     (inc idx#))
              max-val#))
    :sum `(loop [sum-val# (* ~in-alpha (read-datatype-cast-fn
                                        ~datatype
                                        (b-get ~input
                                              (.idx_to_address ~addr ~idx-start))))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (+ sum-val# (* ~in-alpha (read-datatype-cast-fn
                                               ~datatype
                                               (b-get ~input
                                                     (.idx_to_address ~addr idx#)))))
                     (inc idx#))
              sum-val#))
    :mean `(loop [sum-val# (* ~in-alpha
                              (read-datatype-cast-fn
                               ~datatype
                               (b-get ~input
                                     (.idx_to_address ~addr ~idx-start))))
                 idx# (+ ~idx-start 1)]
            (if (< idx# ~idx-stop)
              (recur (+ sum-val# (* ~in-alpha
                                    (read-datatype-cast-fn
                                     ~datatype
                                     (b-get ~input
                                           (.idx_to_address ~addr idx#)))))
                     (inc idx#))
              (/ sum-val#
                 (- ~idx-stop ~idx-start))))
    :magnitude `(loop [sum-val# (square-expr (* ~in-alpha
                                                (read-datatype-cast-fn
                                                 ~datatype
                                                 (b-get ~input (.idx_to_address
                                                               ~addr ~idx-start)))))
                       idx# (+ ~idx-start 1)]
                  (if (< idx# ~idx-stop)
                    (recur (+ sum-val# (square-expr (* ~in-alpha
                                                       (read-datatype-cast-fn
                                                        ~datatype
                                                        (b-get ~input (.idx_to_address
                                                                      ~addr idx#))))))
                           (inc idx#))
                    (Math/sqrt sum-val#)))
    :magnitude-squared `(loop [sum-val# (square-expr (* ~in-alpha
                                                        (read-datatype-cast-fn
                                                         ~datatype
                                                         (b-get ~input
                                                               (.idx_to_address
                                                                ~addr ~idx-start)))))
                               idx# (+ ~idx-start 1)]
                          (if (< idx# ~idx-stop)
                            (recur (+ sum-val# (square-expr
                                                (* ~in-alpha
                                                   (read-datatype-cast-fn
                                                    ~datatype
                                                    (b-get ~input
                                                          (.idx_to_address
                                                           ~addr idx#))))))
                                   (inc idx#))
                            sum-val#))))


(defmacro unary-reduce-impl
  [datatype op]
  `(fn [output# output-dims# input-alpha# input# input-dims#]
     (let [input-shape# (ct-dims/shape input-dims#)
           output-addr# (get-elem-dims->address output-dims#
                                                (ct-dims/shape output-dims#))
           input-addr# (get-elem-dims->address input-dims# (ct-dims/shape input-dims#))
           input# (item->typed-nio-buffer ~datatype input#)
           output# (item->typed-nio-buffer ~datatype output#)
           input-alpha# (datatype->cast-fn ~datatype input-alpha#)
           parallelism# (ct-dims/ecount output-dims#)
           iter-amount# (quot (ct-dims/ecount input-dims#)
                              parallelism#)]
       (parallel/parallel-for
        par-idx# parallelism#
        (let [iter-start# (* par-idx# iter-amount#)
              iter-stop# (+ iter-start# iter-amount#)]
         (b-put output# (.idx_to_address output-addr# par-idx#)
                 (store-datatype-cast-fn ~datatype
                                    (do-unary-reduce-op ~datatype ~op input# input-addr# input-alpha#
                                                        iter-start# iter-stop#))))))))


(defmacro unary-reduce-iter
  []
  (->> (for [dtype all-datatypes
             reduce-op ct/unary-reduction-operations]
         [[dtype reduce-op] {:unary-reduce! `(unary-reduce-impl ~dtype ~reduce-op)}])
       (into {})))


(def ^:private unary-reduce-table
  (unary-reduce-iter))


(defmacro ^:private blas-macro-iter
  [inner-macro]
  `{:float64 (~inner-macro :float64 .dgemm .dgemv)
    :float32 (~inner-macro :float32 .sgemm .sgemv)})


(defmacro ^:private blas-impl
  [datatype gemm-op gemv-op]
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
                  alpha# (datatype->cast-fn ~datatype alpha#)
                  beta# (datatype->cast-fn ~datatype beta#)
                  A# (item->typed-nio-buffer ~datatype A#)
                  B# (item->typed-nio-buffer ~datatype B#)
                  C# (item->typed-nio-buffer ~datatype C#)
                  A-offset# (.position A#)
                  B-offset# (.position B#)
                  C-offset# (.position C#)
                  A# (.array A#)
                  B# (.array B#)
                  C# (.array C#)]
              (when-not (and A# B# C#)
                (throw (ex-info "All nio buffers must be array backed"
                                {:a A#
                                 :b B#
                                 :c C#})))
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
                  A# (item->typed-nio-buffer ~datatype A#)
                  x# (item->typed-nio-buffer ~datatype x#)
                  y# (item->typed-nio-buffer ~datatype y#)
                  A-offset# (.position A#)
                  x-offset# (.position x#)
                  y-offset# (.position y#)
                  A# (.array A#)
                  x# (.array x#)
                  y# (.array y#)
                  alpha# (datatype->cast-fn ~datatype alpha#)
                  inc-x# (long inc-x#)
                  beta# (datatype->cast-fn ~datatype beta#)
                  inc-y# (long inc-y#)]
              (~gemv-op (BLAS/getInstance)
               (cmu/bool->blas-trans trans-a?#)
               a-row-count# a-col-count#
               alpha# A# A-offset# a-rowstride#
               x# x-offset# inc-x#
               beta# y# y-offset# inc-y#)))})


(def ^:private blas-fn-map
  (blas-macro-iter blas-impl))


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
    (let [rand-view (item->typed-nio-buffer :float32 (->buffer dest))
          elem-count (ct-dims/ecount (->dimensions dest))
          rand-gen (SecureRandom.)]
      (cond
        (= (:type distribution) :gaussian)
        (let [mean (float (:mean distribution))
              multiplier (Math/sqrt (float (:variance distribution)))]
          (c-for [idx 0 (< idx elem-count) (inc idx)]
                 (let [next-rand (+ (* multiplier (.nextGaussian rand-gen))
                                    mean)]
                   (b-put rand-view idx next-rand))))
        (= (:type distribution) :flat)
        (let [minimum (float (:minimum distribution))
              maximum (float (:maximum distribution))
              range (- maximum minimum)]
         (c-for [idx 0 (< idx elem-count) (inc idx)]
                (b-put rand-view idx (+ minimum
                                         (* (.nextFloat rand-gen)
                                            range)))))
        :else
        (throw (Exception. (str "Unrecognized distribution: " distribution)))))))


(defn as-tensor
  [java-array]
  (ct/construct-tensor (drv/get-device ct/*stream*)
                       (ct-dims/dimensions [(ct/ecount java-array)])
                       (unsigned/->typed-buffer java-array)))


(defn as-java-array
  [cpu-tensor]
  (drv/sync-with-host ct/*stream*)
  (let [dev-buffer (ct/tensor->buffer cpu-tensor)]
    (dtype/->array dev-buffer)))


(defmacro tensor-context
  [& body]
  `(resource/with-resource-context
     (let [device# (drv/default-device (cpu-driver/driver))
           stream# (drv/create-stream :device device#)]
       (ct/with-stream stream#
         ~@body))))


(defn typed-bufferable->tensor
  "If the typed buffer is not array backend then gemm, gemv will not work."
  [item]
  (let [typed-buffer (unsigned/->typed-buffer item)]
    (ct/construct-tensor (ct-dims/dimensions (ct/shape item))
                         typed-buffer)))
