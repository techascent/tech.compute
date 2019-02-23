(ns tech.compute.cpu.tensor-math.unary-reduce
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
            [tech.compute.tensor.dimensions :as ct-dims]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.tensor :as ct])
  (:import [tech.compute.cpu UnaryReduce]))


(defmacro perform-reduce
  [init-val update-val finalize-val
  datatype input addr in-alpha idx-start idx-stop]
  `(loop [sum-val# (~init-val (* ~in-alpha
                                 (read-datatype-cast-fn
                                  ~datatype
                                  (b-get ~input
                                         (.idx_to_address ~addr ~idx-start)))))
          idx# (+ ~idx-start 1)]
     (if (< idx# ~idx-stop)
       (recur (~update-val sum-val# (* ~in-alpha
                                       (read-datatype-cast-fn
                                        ~datatype
                                        (b-get ~input
                                               (.idx_to_address ~addr idx#)))))
              (inc idx#))
       (~finalize-val sum-val# (- ~idx-stop ~idx-start)))))


(defmacro custom-init
  [arg]
  `(.initialize ~'custom (double ~arg)))


(defmacro custom-update
  [acc next]
  `(.update ~'custom ~acc (double ~next)))


(defmacro custom-finalize
  [acc idx-range]
  `(.finalize ~'custom (double ~acc) (int ~idx-range)))


(defmacro custom-unary-reduce-impl
  [datatype]
  `(fn [output# output-dims# input-alpha# input# input-dims# unary-op#]
     (let [input-shape# (ct-dims/shape input-dims#)
           output-addr# (get-elem-dims->address output-dims#
                                                (ct-dims/shape output-dims#))
           input-addr# (get-elem-dims->address input-dims# (ct-dims/shape input-dims#))
           input# (item->typed-nio-buffer ~datatype input#)
           output# (item->typed-nio-buffer ~datatype output#)
           input-alpha# (datatype->cast-fn ~datatype input-alpha#)
           parallelism# (ct-dims/ecount output-dims#)
           iter-amount# (quot (ct-dims/ecount input-dims#)
                              parallelism#)
           ^UnaryReduce ~'custom unary-op#]
       (parallel/parallel-for
        par-idx# parallelism#
        (let [iter-start# (* par-idx# iter-amount#)
              iter-stop# (+ iter-start# iter-amount#)]
         (b-put output# (.idx_to_address output-addr# par-idx#)
                (store-datatype-cast-fn
                 ~datatype
                 (perform-reduce custom-init custom-update custom-finalize
                                 ~datatype input# input-addr# input-alpha#
                                 iter-start# iter-stop#))))))))


(defmacro make-custom-unary-reducers
  []
  (->> (for [dtype all-datatypes]
         [[dtype :custom] {:unary-reduce! `(custom-unary-reduce-impl ~dtype)}])
       (into {})))


(def custom-unary-reducers (make-custom-unary-reducers))


(def unary-reduce-table custom-unary-reducers)
