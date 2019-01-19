(ns tech.compute.cpu.jna-lapack
  "These are fortran functions.  So you have to call them in fortran style."
  (:require [tech.datatype :as dtype]
            [tech.jna :as jna]
            [tech.compute.cpu.jna-blas :as jna-blas]
            [tech.datatype.java-primitive :as primitive]))


(def ^:dynamic *system-lapack-name* "lapack")


(defn int-out-value
  [arg]
  (when-not (instance? (Class/forName "[I") arg)
    (throw (ex-info "Item is not an int-array" {:argument arg})))
  (when-not (>= (dtype/ecount arg) 1)
    (throw (ex-info "Out parameters must have lenghth >= 1"
                    {:item-len (dtype/ecount arg)})))
  arg)


(defn int-in-value
  [arg]
  (if (instance? (Class/forName "[I") arg)
    (int-out-value arg)
    (int-array [(int arg)])))


(defn int-ary-in-value
  [arg]
  (int-out-value arg))


(defn upload-str
  [upload]
  (if-let [retval (#{"U" "L"} (str upload))]
    retval
    (throw (ex-info "Invalid upload type" {:upload upload}))))


;;;; Cholesky

;;;; Floating point

(jna/def-jna-fn *system-lapack-name* spotrf_
  "Lapack Cholesky decomposition"
  nil
  [upload upload-str]
  [N int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [info int-out-value])


(jna/def-jna-fn *system-lapack-name* spotrs_
  "Cholesky solve Ax=B system.  A has been cholesky decomposed."
  nil
  [upload upload-str]
  [N int-in-value]
  [NRHS int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [B primitive/ensure-ptr-like]
  [ldb int-in-value]
  [info int-out-value])


;;Double

(jna/def-jna-fn *system-lapack-name* dpotrf_
  "Lapack Cholesky decomposition"
  nil
  [upload upload-str]
  [N int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [info int-out-value])


(jna/def-jna-fn *system-lapack-name* dpotrs_
  "Cholesky solve Ax=B system.  A has been cholesky decomposed."
  nil
  [upload upload-str]
  [N int-in-value]
  [NRHS int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [B primitive/ensure-ptr-like]
  [ldb int-in-value]
  [info int-out-value])


;;;; LU
(jna/def-jna-fn *system-lapack-name* sgetrf_
  "LU factorize matrix."
  nil
  [M int-in-value]
  [N int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [ipiv primitive/ensure-ptr-like]
  [info int-out-value])


(jna/def-jna-fn *system-lapack-name* sgetrf_
  "LU factorize float matrix."
  nil
  [M int-in-value]
  [N int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [ipiv primitive/ensure-ptr-like]
  [info int-out-value])


(jna/def-jna-fn *system-lapack-name* dgetrf_
  "LU factorize double matrix."
  nil
  [M int-in-value]
  [N int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [ipiv primitive/ensure-ptr-like]
  [info int-out-value])


(defn trans-str
  [arg]
  (if-let [retval (#{"N" "T" "C"} (str arg))]
    retval
    (throw (ex-info "Argument is not a transpose str"
                    {:arg arg
                     :trans-str-set #{"N" "T" "C"}}))))


(jna/def-jna-fn *system-lapack-name* sgetrs_
  "LU solve float matrix."
  nil
  [trans trans-str]
  [N int-in-value]
  [NRHS int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [ipiv int-ary-in-value]
  [B primitive/ensure-ptr-like]
  [ldb int-in-value]
  [info int-out-value])


(jna/def-jna-fn *system-lapack-name* dgetrs_
  "LU solve double matrix."
  nil
  [trans trans-str]
  [N int-in-value]
  [NRHS int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [ipiv int-ary-in-value]
  [B primitive/ensure-ptr-like]
  [ldb int-in-value]
  [info int-out-value])



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
     :retval (vec retval)}))
