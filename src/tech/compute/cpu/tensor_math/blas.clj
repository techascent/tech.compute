(ns tech.compute.cpu.tensor-math.blas
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
            [tech.compute.math-util :as cmu])
    (:import [com.github.fommil.netlib BLAS]))


(defmacro ^:private blas-macro-iter
  [inner-macro]
  `{:float64 (~inner-macro :float64 .dgemm)
    :float32 (~inner-macro :float32 .sgemm)})


(defmacro ^:private blas-impl
  [datatype gemm-op]
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
               beta# C# C-offset# c-rowstride#)))})


(def ^:private blas-fn-map
  (blas-macro-iter blas-impl))
