(ns tech.compute.cpu.tensor-math.unary-op
  (:require [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     all-datatypes
                     ] :as nio-access]
            [tech.compute.cpu.tensor-math.addressing
             :refer [get-elem-dims->address
                     max-shape-from-dimensions]]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math.writers :as writers])
  (:import [tech.compute.cpu UnaryOp]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defmacro create-unary-op
  [op-code]
  `(reify UnaryOp
     (op [this# ~'x]
       (double ~op-code))))


(def builtin-unary-ops
  {:floor (create-unary-op (Math/floor x))
   :ceil (create-unary-op (Math/ceil x))
   :round (create-unary-op (Math/round x))
   :- (create-unary-op (- x))
   :tanh (create-unary-op (Math/tanh x))
   :logistic (create-unary-op
              (/ 1.0
                 (+ 1.0 (Math/exp (- x)))))
   :exp (create-unary-op (Math/exp x))
   :sqrt (create-unary-op (Math/sqrt x))
   :noop (create-unary-op x)})


(defmacro ^:private custom-accum!-impl
  [datatype]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims# dest-alpha#
          n-elems# unary-op#]
       (let [n-elems# (long n-elems#)
             dest# (item->typed-nio-buffer ~datatype dest#)
             dest-idx->address# (get-elem-dims->address dest-dims#
                                                        (get dest-dims# :shape))
             max-shape# (max-shape-from-dimensions dest-dims#)
             dest-alpha# (unsigned/datatype->cast-fn :ignored ~datatype dest-alpha#)
             ^UnaryOp custom# unary-op#
             writer# (writers/get-serial-writer ~jvm-datatype)]
         (writer# dest# dest-dims# max-shape# n-elems#
                  (primitive/make-converter
                   ~jvm-datatype
                   (store-datatype-cast-fn
                    ~datatype
                    (.op custom#
                         (*
                          (read-datatype-cast-fn
                           ~datatype
                           (b-get dest# (.idx_to_address dest-idx->address#
                                                         ~'idx)))
                          dest-alpha#)))))))))


(defmacro ^:private custom-unary-op!-impl
  [datatype]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims#
          x# x-dims# x-alpha#
          n-elems# unary-op#]
       (let [n-elems# (long n-elems#)
             max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
             x# (item->typed-nio-buffer ~datatype x#)
             x-idx->address# (get-elem-dims->address x-dims# max-shape#)
             x-alpha# (unsigned/datatype->cast-fn :ignored ~datatype x-alpha#)
             ^UnaryOp custom-op# unary-op#
             writer# (writers/get-parallel-writer ~jvm-datatype)]
         (writer# dest# dest-dims# max-shape# n-elems#
                  (primitive/make-converter
                   ~jvm-datatype
                   (store-datatype-cast-fn
                    ~datatype
                    (.op custom-op#
                         (* (read-datatype-cast-fn
                             ~datatype
                             (b-get x# (.idx_to_address x-idx->address# ~'idx)))
                            x-alpha#)))))))))


(defmacro make-custom-unary-ops
  []
  (->> (for [dtype all-datatypes]
         [[dtype :custom] {:unary-accum! `(custom-accum!-impl ~dtype)
                           :unary-op! `(custom-unary-op!-impl ~dtype)}])
       (into {})))


(def custom-unary-ops (make-custom-unary-ops))


(def unary-op-table
  (merge custom-unary-ops
         (->> (for [dtype all-datatypes
                    [op-name operator] builtin-unary-ops]
                (let [{:keys [unary-accum! unary-op!]}
                      (get custom-unary-ops [dtype :custom])]
                  [[dtype op-name]
                   {:unary-op!
                    (fn [dest dest-dims
                         x x-dims x-alpha
                         n-elems]
                      (unary-op! dest dest-dims x x-dims x-alpha
                                 n-elems operator))
                    :unary-accum!
                    (fn [dest dest-dims dest-alpha
                         n-elems]
                      (unary-accum! dest dest-dims dest-alpha
                                    n-elems operator))}]))
              (into {}))))
