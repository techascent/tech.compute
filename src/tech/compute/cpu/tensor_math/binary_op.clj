(ns tech.compute.cpu.tensor-math.binary-op
  (:require [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     all-datatypes
                     datatype->cast-fn
                     binary-op-rev-ops
                     binary-op-impl
                     ] :as nio-access]
            [tech.compute.cpu.tensor-math.addressing
             :refer [get-elem-dims->address
                     max-shape-from-dimensions]]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.tensor :as ct])
  (:import [tech.compute.cpu BinaryOp]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


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
                         (binary-op-rev-ops ~operation ~reverse-operands?
                                            (* (read-datatype-cast-fn
                                                ~datatype
                                                (b-get x# x-idx#)) x-alpha#)
                                            scalar#))))))))


(defmacro ^:private custom-binary-op-constant!-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        scalar#
        n-elems#
        custom-op#
        reverse-operands?#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           dest# (item->typed-nio-buffer ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (item->typed-nio-buffer ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           scalar# (double (datatype->cast-fn ~datatype scalar#))
           ^BinaryOp custom-op# custom-op#]
       (if reverse-operands?#
         (parallel/parallel-for
          idx# (long n-elems#)
          (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
                x-idx# (.idx_to_address x-idx->address# idx#)]
            (b-put dest# dest-idx#
                   (store-datatype-cast-fn
                    ~datatype
                    (.op custom-op#
                         scalar#
                         (* (read-datatype-cast-fn
                             ~datatype
                             (b-get x# x-idx#)) x-alpha#))))))
         (parallel/parallel-for
          idx# (long n-elems#)
          (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
                x-idx# (.idx_to_address x-idx->address# idx#)]
            (b-put dest# dest-idx#
                   (store-datatype-cast-fn
                    ~datatype
                    (.op custom-op#
                         (* (read-datatype-cast-fn
                             ~datatype
                             (b-get x# x-idx#)) x-alpha#)
                         scalar#)))))))))


(defmacro binary-op-constant-table
  []
  (->> (concat (for [dtype all-datatypes
                     op ct/binary-operations
                     rev-ops? [true false]]
                 [[dtype op rev-ops?] `(binary-op-constant!-impl ~dtype ~op ~rev-ops?)])
               (for [dtype all-datatypes]
                 [[dtype :custom] `(custom-binary-op-constant!-impl ~dtype)]
                 ))
       (into {})))


(def binary-op-constant-table
  (binary-op-constant-table))


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
                         (binary-op-impl ~operation
                                         (* (read-datatype-cast-fn
                                             ~datatype
                                             (b-get x# x-idx#)) x-alpha#)
                                         (* (read-datatype-cast-fn
                                             ~datatype
                                             (b-get y# y-idx#)) y-alpha#)))))))))


(defmacro ^:private custom-binary-op!-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        y# y-dims# y-alpha#
        n-elems#
        custom-op#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims# y-dims#)
           dest# (item->typed-nio-buffer ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (item->typed-nio-buffer ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (datatype->cast-fn ~datatype x-alpha#)
           y# (item->typed-nio-buffer ~datatype y#)
           y-idx->address# (get-elem-dims->address y-dims# max-shape#)
           y-alpha# (datatype->cast-fn ~datatype y-alpha#)
           ^BinaryOp custom-op# custom-op#]
       (parallel/parallel-for
        idx# (long n-elems#)
        (let [dest-idx# (.idx_to_address dest-idx->address# idx#)
              x-idx# (.idx_to_address x-idx->address# idx#)
              y-idx# (.idx_to_address y-idx->address# idx#)]
          (b-put dest# dest-idx#
                        (store-datatype-cast-fn
                         ~datatype
                         (.op custom-op#
                              (* (read-datatype-cast-fn
                                  ~datatype
                                  (b-get x# x-idx#)) x-alpha#)
                              (* (read-datatype-cast-fn
                                  ~datatype
                                  (b-get y# y-idx#)) y-alpha#)))))))))


(defmacro binary-op-table-impl
  []
  (->> (concat (for [dtype all-datatypes
                     op ct/binary-operations]
                 [[dtype op] `(binary-op!-impl ~dtype ~op)])
               (for [dtype all-datatypes]
                 [[dtype :custom] `(custom-binary-op!-impl ~dtype)]
                 ))
       (into {})))


(def ^:private binary-op-table
  (binary-op-table-impl))
