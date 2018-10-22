(ns tech.compute.cpu.tensor-math.binary-accum
  (:require [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     all-datatypes
                     datatype->cast-fn
                     binary-op-rev-ops
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


(defmacro ^:private binary-accum-constant!-impl
  [datatype operation reverse-operands?]
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
                       (store-datatype-cast-fn
                        ~datatype
                        (binary-op-rev-ops ~operation ~reverse-operands?
                                            (* (read-datatype-cast-fn
                                                ~datatype
                                                (b-get dest# dest-idx#)) dest-alpha#)
                                            scalar#))))))))


(defmacro binary-accum-constant-table
  []
  (->> (for [dtype all-datatypes
             op ct/binary-operations
             rev-ops? [true false]]
         [[dtype op rev-ops?] `(binary-accum-constant!-impl ~dtype ~op ~rev-ops?)])
       (into {})))


(def binary-accum-constant-table
  (binary-accum-constant-table))


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
                               (binary-op-rev-ops ~operation ~reverse-operands?
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


(def binary-accum-table
  (binary-accum-table))
