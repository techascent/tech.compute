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
  `{:float64 (~inner-macro :float64 .dgemm .dgemv)
    :float32 (~inner-macro :float32 .sgemm .sgemv)})


(defmacro ^:private blas-impl
  [datatype gemm-op gemv-op]
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
               beta# C# C-offset# c-rowstride#)))
    :gemv (fn [trans-a?# a-row-count# a-col-count#
               alpha# A# a-rowstride#
               x# inc-x#
               beta# y# inc-y#]
            (let [a-rowstride# (long a-rowstride#)
                  a-row-count# (long a-row-count#)
                  a-col-count# (long a-col-count#)
                  A# (item->typed-nio-buffer ~datatype A#)
                  x# (item->typed-nio-buffer ~datatype x#)
                  y# (item->typed-nio-buffer ~datatype y#)
                  A-offset# (.position A#)
                  x-offset# (.position x#)
                  y-offset# (.position y#)
                  A# (.array A#)
                  x# (.array x#)
                  y# (.array y#)
                  alpha# (datatype->cast-fn ~datatype alpha#)
                  inc-x# (long inc-x#)
                  beta# (datatype->cast-fn ~datatype beta#)
                  inc-y# (long inc-y#)]
              (~gemv-op (BLAS/getInstance)
               (cmu/bool->blas-trans trans-a?#)
               a-row-count# a-col-count#
               alpha# A# A-offset# a-rowstride#
               x# x-offset# inc-x#
               beta# y# y-offset# inc-y#)))})


(def ^:private blas-fn-map
  (blas-macro-iter blas-impl))
