(ns tech.compute.tensor.lapack
  (:require [tech.compute.tensor :as ct]
            [tech.compute.tensor.math :as tm]
            [tech.compute.tensor.dimensions :as dims]
            [tech.compute.tensor.defaults :as defaults]
            [tech.compute.tensor.error-checking :as error-checking]
            [tech.compute.tensor.details :as details]
            [tech.compute.tensor.utils :refer [when-not-error]]
            [tech.datatype :as dtype]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn cholesky-factorize!
  "dpotrf bindings.  Dest is both input and result argument.  dest: io argument,
  corresponding to matrix that is being factorized.  Results are stored in A.
  upload: :upper or :lower, store U or L."
  [dest-A upload & {:as options}]
  (let [dest-A (ct/ensure-tensor dest-A)
        A-dims (ct/tensor->dimensions dest-A)
        a-in-place-trans? (not (dims/access-increasing? A-dims))
        [a-rows a-cols] (dims/shape A-dims)
        a-rows (long a-rows)
        a-cols (long a-cols)
        stream (defaults/infer-stream options dest-A)]
    (error-checking/ensure-cudnn-datatype (dtype/get-datatype dest-A) :cholesky-factorize!)
    (error-checking/ensure-matrix dest-A)
    (when-not-error (= a-rows a-cols)
      "Cholesky factorization only defined for symmetric nxn matrixes"
      {:n-rows a-rows
       :n-cols a-cols})
    (when-not-error (#{:lower :upper} upload)
      "Cholesky upload parameter must be either :lower or :upper"
      {:upload-param upload})
    (let [upload (if a-in-place-trans?
                   (if (= :lower upload)
                     :upper
                     :lower)
                   upload)]
      (tm/cholesky-factorize! stream dest-A upload)
      dest-A)))


(defn cholesky-solve!
  "Cholesky solve a system of equations.

  dest-B: Matrix to solve.
  upload: :upper or :lower depending on if A is upper or lower.
  A: cholesky-decomposed matrix A."
  [dest-B upload A & {:as options}]
  (let [dest-B (ct/ensure-tensor dest-B)
        A (ct/ensure-tensor A)
        stream (defaults/infer-stream options dest-B)
        A-dims (ct/tensor->dimensions A)
        a-in-place-trans? (not (dims/access-increasing? A-dims))
        [a-rows a-cols] (dims/shape A-dims)
        a-rows (long a-rows)
        a-cols (long a-cols)]
    (error-checking/ensure-cudnn-datatype (dtype/get-datatype dest-B) :cholesky-solve!)
    (error-checking/ensure-datatypes (dtype/get-datatype dest-B) A)
    (when-not-error (= a-rows a-cols)
      "Cholesky factorization only defined for symmetric nxn matrixes"
      {:n-rows a-rows
       :n-cols a-cols})
    (when-not-error (#{:lower :upper} upload)
      "Cholesky upload parameter must be either :lower or :upper"
      {:upload-param upload})
    (let [upload (if a-in-place-trans?
                   (if (= :lower upload)
                     :upper
                     :lower)
                   upload)]
      (tm/cholesky-solve! stream dest-B upload A)
      dest-B)))



(defn LU-factorize!
  "This returns both A and the integer pivot table.
  retuns:
  {:LU LU matrix written into A
   :pivots pivots}"
  [dest-A & {:keys [ipiv]
             :as options}]
  (let [dest-A (ct/ensure-tensor dest-A)
        _ (error-checking/ensure-cudnn-datatype (dtype/get-datatype dest-A) :LU-factorize!)
        _ (error-checking/ensure-matrix dest-A)
        [a-row-count a-col-count] (ct/shape dest-A)
        stream (defaults/infer-stream options dest-A)
        ipiv (if ipiv
               ipiv
               (ct/new-tensor [(max (int a-row-count) (int a-col-count))] :init-value nil
                              :stream stream
                              :datatype :int32))]
    (when-not-error (= :int32 (dtype/get-datatype ipiv))
      "Pivot table must be int32 datatype"
      {:pivot-datatype (dtype/get-datatype ipiv)})
    (tm/LU-factorize! stream dest-A (ct/ensure-tensor ipiv))
    {:LU dest-A
     :pivots ipiv}))


(def trans-cmd-set #{:no-transpose :transpose :conjugate-transpose})


(defn LU-solve!
  "Solve a matrix using an LU factored system"
  [dest-B trans-cmd A pivots & options]
  (let [dest-B (ct/ensure-tensor dest-B)
        A (ct/ensure-tensor A)
        pivots (ct/ensure-tensor pivots)
        stream (defaults/infer-stream options dest-B A pivots)]
    (when-not-error (= :int32 (dtype/get-datatype pivots))
      "Pivot table must be int32 datatype"
      {:pivot-datatype (dtype/get-datatype pivots)})
    (when-not-error (trans-cmd-set trans-cmd)
      "Transpose cmd must be in trans cmd set"
      {:trans-cmd trans-cmd
       :trans-cmd-set trans-cmd-set})
    (error-checking/ensure-datatypes (dtype/get-datatype dest-B) A)
    (error-checking/ensure-cudnn-datatype (dtype/get-datatype dest-B) :LU-solve!)
    (error-checking/ensure-matrix dest-B)
    (error-checking/ensure-matrix A)
    (error-checking/ensure-vector pivots)
    (error-checking/ensure-same-device dest-B A pivots)
    (error-checking/ensure-simple-tensor pivots)
    (when-not-error (= 1 (count (ct/shape pivots)))
      "Pivots must be dense simple tensor" {})
    (let [[a-row-count a-col-count] (ct/shape A)]
      (when-not-error (= (max (int a-row-count) (int a-col-count))
                         (ct/ecount pivots))
        "Pivots must be max(n-rows,ncols)"
        {:pivot-shape (ct/shape pivots)})
      (tm/LU-solve! stream dest-B trans-cmd A pivots)
      dest-B)))
