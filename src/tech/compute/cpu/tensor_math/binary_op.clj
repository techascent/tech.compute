(ns tech.compute.cpu.tensor-math.binary-op
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
            [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math.writers :as writers]
            [tech.compute.cpu.tensor-math.binary-op-impls :as bin-impls])
  (:import [tech.compute.cpu BinaryOp]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defmacro ^:private custom-binary-op-constant!-impl
  [datatype]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims#
          x# x-dims# x-alpha#
          scalar#
          n-elems#
          custom-op#
          reverse-operands?#]
       (let [n-elems# (long n-elems#)
             max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
             x# (item->typed-nio-buffer ~datatype x#)
             x-idx->address# (get-elem-dims->address x-dims# max-shape#)
             x-alpha# (datatype->cast-fn ~datatype x-alpha#)
             scalar# (double (datatype->cast-fn ~datatype scalar#))
             writer# (writers/get-parallel-writer ~jvm-datatype)
             converter# (bin-impls/make-converter-elide-modulators
                         ~datatype reverse-operands?# custom-op#
                         x-alpha# (datatype->cast-fn ~datatype 1)
                         (read-datatype-cast-fn
                          ~datatype
                          (b-get x# (.idx_to_address x-idx->address# ~'idx)))
                         scalar#)]
         (writer# dest# dest-dims# max-shape# n-elems# converter#)))))


(defmacro make-custom-bin-op-constant-table
  []
  (->> (for [dtype all-datatypes]
         [[dtype :custom] `(custom-binary-op-constant!-impl ~dtype)])
       (into {})))


(def custom-bin-op-constant-table (make-custom-bin-op-constant-table))


(def binary-op-constant-table custom-bin-op-constant-table)


(defmacro ^:private custom-binary-op!-impl
  [datatype]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims#
          x# x-dims# x-alpha#
          y# y-dims# y-alpha#
          n-elems#
          custom-op#]
       (let [n-elems# (long n-elems#)
             max-shape# (max-shape-from-dimensions dest-dims# x-dims# y-dims#)
             x# (item->typed-nio-buffer ~datatype x#)
             x-idx->address# (get-elem-dims->address x-dims# max-shape#)
             x-alpha# (datatype->cast-fn ~datatype x-alpha#)
             y# (item->typed-nio-buffer ~datatype y#)
             y-idx->address# (get-elem-dims->address y-dims# max-shape#)
             y-alpha# (datatype->cast-fn ~datatype y-alpha#)
             writer# (writers/get-parallel-writer ~jvm-datatype)
             converter# (bin-impls/make-converter-elide-modulators
                         ~datatype false custom-op#
                         x-alpha# y-alpha#
                         (read-datatype-cast-fn
                          ~datatype
                          (b-get x# (.idx_to_address x-idx->address# ~'idx)))
                         (read-datatype-cast-fn
                          ~datatype
                          (b-get y# (.idx_to_address y-idx->address# ~'idx))))]
         (writer# dest# dest-dims# max-shape# n-elems# converter#)))))


(defmacro make-custom-bin-op-table
  []
  (->> (for [dtype all-datatypes]
         [[dtype :custom] `(custom-binary-op!-impl ~dtype)])
       (into {})))


(def custom-bin-op-table (make-custom-bin-op-table))


(def binary-op-table custom-bin-op-table)
