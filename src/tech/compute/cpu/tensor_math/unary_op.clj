(ns tech.compute.cpu.tensor-math.unary-op
  (:require [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     all-datatypes
                     unary-op-impl
                     ] :as nio-access]
            [tech.compute.cpu.tensor-math.addressing
             :refer [get-elem-dims->address
                     max-shape-from-dimensions]]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.tensor :as ct])
  (:import [tech.compute.cpu UnaryOp]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defmacro ^:private unary-accum!-impl
  [datatype operation]
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
                       (store-datatype-cast-fn
                        ~datatype
                        (unary-op-impl
                         ~operation
                         (*
                          (read-datatype-cast-fn
                           ~datatype
                           (b-get dest# dest-idx#))
                          dest-alpha#)))))))))


(defmacro ^:private custom-accum!-impl
  [datatype]
  `(fn [dest# dest-dims# dest-alpha#
        n-elems# unary-op#]
     (let [n-elems# (long n-elems#)
           dest# (item->typed-nio-buffer ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims#
                                                      (get dest-dims# :shape))
           dest-alpha# (unsigned/datatype->cast-fn :ignored ~datatype dest-alpha#)
           ^UnaryOp custom# unary-op#]
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [dest-idx# (.idx_to_address dest-idx->address# idx#)]
                (b-put dest# dest-idx#
                       (store-datatype-cast-fn
                        ~datatype
                        (.op custom#
                         (*
                          (read-datatype-cast-fn
                           ~datatype
                           (b-get dest# dest-idx#))
                          dest-alpha#)))))))))


(defmacro ^:private unary-op!-impl
  [datatype operation]
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
               (store-datatype-cast-fn
                ~datatype
                (unary-op-impl
                 ~operation
                 (* (read-datatype-cast-fn
                     ~datatype
                     (b-get x# (.idx_to_address x-idx->address# idx#)))
                    x-alpha#))))))))


(defmacro ^:private custom-unary-op!-impl
  [datatype]
  `(fn [dest# dest-dims#
        x# x-dims# x-alpha#
        n-elems# unary-op#]
     (let [n-elems# (long n-elems#)
           max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
           dest# (item->typed-nio-buffer ~datatype dest#)
           dest-idx->address# (get-elem-dims->address dest-dims# max-shape#)
           x# (item->typed-nio-buffer ~datatype x#)
           x-idx->address# (get-elem-dims->address x-dims# max-shape#)
           x-alpha# (unsigned/datatype->cast-fn :ignored ~datatype x-alpha#)
           n-elems# (long n-elems#)
           ^UnaryOp custom-op# unary-op#]
       (parallel/parallel-for
        idx# n-elems#
        (b-put dest# (.idx_to_address dest-idx->address# idx#)
               (store-datatype-cast-fn
                ~datatype
                (.op custom-op#
                 (* (read-datatype-cast-fn
                     ~datatype
                     (b-get x# (.idx_to_address x-idx->address# idx#)))
                    x-alpha#))))))))


(defmacro unary-op-table-impl
  []
  (->> (concat (for [dtype all-datatypes
                     op ct/unary-operations]
                 [[dtype op] {:unary-accum! `(unary-accum!-impl ~dtype ~op)
                              :unary-op! `(unary-op!-impl ~dtype ~op)}])
               (for [dtype all-datatypes]
                 [[dtype :custom] {:unary-accum! `(custom-accum!-impl ~dtype)
                                   :unary-op! `(custom-unary-op!-impl ~dtype)}]))
       (into {})))


(def unary-op-table
  (unary-op-table-impl))
