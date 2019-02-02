(ns tech.compute.cpu.tensor-math.writers
  (:require [tech.datatype.java-primitive :as primitive]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put item->typed-nio-buffer]
             :as nio-access]
            [tech.compute.cpu.tensor-math.addressing
             :refer [get-elem-dims->address]]
            [tech.parallel :as parallel]
            [clojure.core.matrix.macros :refer [c-for]])
  (:import  [tech.datatype ByteConverter ShortConverter IntConverter
             LongConverter FloatConverter DoubleConverter]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defmacro parallel-writer-fn
  [dst-dtype]
  `(fn [dst# dst-dims# max-shape# n-elems# converter#]
     (let [dest# (item->typed-nio-buffer ~dst-dtype dst#)
           dest-idx->address# (get-elem-dims->address dst-dims# max-shape#)
           n-elems# (int n-elems#)
           converter# (primitive/datatype->converter ~dst-dtype converter#)]
       (parallel/parallel-for
        idx# n-elems#
        (b-put dest# (.idx_to_address dest-idx->address# idx#)
               (.convert converter# idx#))))))


(defmacro make-parallel-writers-table
  []
  (->> (for [dtype primitive/datatypes]
         [dtype `(parallel-writer-fn ~dtype)])
       (into {})))


(def parallel-writers-table (make-parallel-writers-table))


(defn get-parallel-writer
  [dtype]
  (if-let [retval (get parallel-writers-table dtype)]
    retval
    (throw (ex-info (format "Failed to find parallel writer for datatype: %s" dtype)
                    {:datatype dtype}))))


(defmacro serial-writer-fn
  [dst-dtype]
  `(fn [dst# dst-dims# max-shape# n-elems# converter#]
     (let [dest# (item->typed-nio-buffer ~dst-dtype dst#)
           dest-idx->address# (get-elem-dims->address dst-dims# max-shape#)
           n-elems# (int n-elems#)
           converter# (primitive/datatype->converter ~dst-dtype converter#)]
       (c-for [idx# (int 0) (< idx# n-elems#) (unchecked-add idx# 1)]
              (b-put dest# (.idx_to_address dest-idx->address# idx#)
                     (.convert converter# idx#))))))


(defmacro make-serial-writers-table
  []
  (->> (for [dtype primitive/datatypes]
         [dtype `(serial-writer-fn ~dtype)])
       (into {})))


(def serial-writers-table (make-serial-writers-table))


(defn get-serial-writer
  [dtype]
  (if-let [retval (get serial-writers-table dtype)]
    retval
    (throw (ex-info (format "Failed to find serial writer for datatype: %s" dtype)
                    {:datatype dtype}))))
