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
            [tech.compute.tensor :as ct]))


(defmacro square-expr
  [expr]
  `(let [item# ~expr]
     (* item# item#)))


(defmacro identity-expr
  [& args]
  (first args))


(defmacro sqrt-expr
  [& args]
  `(Math/sqrt ~(first args)))


(defmacro add-squared
  [sum-var item]
  `(+ ~sum-var (square-expr ~item)))


(defmacro perform-reduce
  [init-val update-val finalize-val
  datatype op input addr in-alpha idx-start idx-stop]
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


(defmacro do-unary-reduce-op
  [datatype op input addr in-alpha idx-start idx-stop]
  (condp = op
    :min `(perform-reduce identity-expr min identity-expr
                          ~datatype ~op ~input ~addr ~in-alpha ~idx-start ~idx-stop)
    :max `(perform-reduce identity-expr max identity-expr
                          ~datatype ~op ~input ~addr ~in-alpha ~idx-start ~idx-stop)
    :sum `(perform-reduce identity-expr + identity-expr
                          ~datatype ~op ~input ~addr ~in-alpha ~idx-start ~idx-stop)
    :mean `(perform-reduce identity-expr + /
                           ~datatype ~op ~input ~addr ~in-alpha ~idx-start ~idx-stop)
    :magnitude `(perform-reduce square-expr add-squared sqrt-expr
                                ~datatype ~op ~input ~addr ~in-alpha ~idx-start ~idx-stop)
    :magnitude-squared `(perform-reduce square-expr add-squared identity-expr
                                        ~datatype ~op ~input ~addr ~in-alpha ~idx-start ~idx-stop)))


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


(def unary-reduce-table
  (unary-reduce-iter))
