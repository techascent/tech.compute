(ns tech.compute.cpu.tensor-math.ternary-op
  (:require [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     all-datatypes
                     datatype->cast-fn
                     ] :as nio-access]
            [tech.compute.cpu.tensor-math.addressing
             :refer [get-elem-dims->address
                     max-shape-from-dimensions]]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.tensor :as ct]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


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
