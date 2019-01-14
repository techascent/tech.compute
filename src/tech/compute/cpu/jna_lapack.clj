(ns tech.compute.cpu.jna-lapack
  "These are fortran functions.  So you have to call them in fortran style."
  (:require [tech.datatype :as dtype]
            [tech.jna :as jna]
            [tech.compute.cpu.jna-blas :as jna-blas]
            [tech.datatype.java-primitive :as primitive]))


(def ^:dynamic *system-lapack-name* "lapack")


;;Floating point

(jna/def-jna-fn *system-lapack-name* spotrf_
  "Lapack Cholesky decomposition"
  nil
  [upload str]
  [N primitive/ensure-ptr-like]
  [A primitive/ensure-ptr-like]
  [lda primitive/ensure-ptr-like]
  [info primitive/ensure-ptr-like])


(jna/def-jna-fn *system-lapack-name* spotrs_
  "Cholesky solve Ax=B system.  A has been cholesky decomposed."
  nil
  [upload str]
  [N primitive/ensure-ptr-like]
  [NRHS primitive/ensure-ptr-like]
  [A primitive/ensure-ptr-like]
  [lda primitive/ensure-ptr-like]
  [B primitive/ensure-ptr-like]
  [ldb primitive/ensure-ptr-like]
  [info primitive/ensure-ptr-like])


;;Double


(jna/def-jna-fn *system-lapack-name* dpotrf_
  "Lapack Cholesky decomposition"
  nil
  [upload str]
  [N primitive/ensure-ptr-like]
  [A primitive/ensure-ptr-like]
  [lda primitive/ensure-ptr-like]
  [info primitive/ensure-ptr-like])


(jna/def-jna-fn *system-lapack-name* dpotrs_
  "Cholesky solve Ax=B system.  A has been cholesky decomposed."
  nil
  [upload str]
  [N primitive/ensure-ptr-like]
  [NRHS primitive/ensure-ptr-like]
  [A primitive/ensure-ptr-like]
  [lda primitive/ensure-ptr-like]
  [B primitive/ensure-ptr-like]
  [ldb primitive/ensure-ptr-like]
  [info primitive/ensure-ptr-like])


(comment
  (let [upload "U"
        N (int-array [3])
        A (double-array [4 1 -1
                         1 2 1
                         -1 1 2])
        lda (int-array [3])
        retval (int-array 1)]
    (dpotrf_ upload N A lda retval)
    {:A (vec A)
     :retval (vec retval)})


  (let [A (double-array [4 1 -1 1 2 1 -1 1 2])
        chol-A (double-array [2.0 1.0 -1.0 0.5 1.3228756555322954 1.0 -0.5 0.944911182523068 0.9258200997725515])
        B (double-array [1 0 0 0 1 0 0 0 1])
        retval (int-array 1)]
    (dpotrs_ "U" (int-array [3]) (int-array [3]) chol-A (int-array [3]) B (int-array [3]) retval)
    {:inv-chol (vec B)
     :chol (vec chol-A)
     :A (vec A)
     :retval (vec retval)})
  )
