(ns tech.compute.cpu.jna-lapack
  "These are fortran functions.  So you have to call them in fortran style."
  (:require [tech.datatype :as dtype]
            [tech.jna :as jna]
            [tech.compute.cpu.jna-blas :as jna-blas]
            [tech.datatype.java-primitive :as primitive]))


(def ^:dynamic *system-lapack-name* "lapack")


(defmacro def-lapack-fn
  [fn-name description & arglist]
  `(do
     (jna/def-jna-fn *system-lapack-name* ~(symbol (str "s" fn-name "_"))
       ~description
       nil
       ~@arglist)
     (jna/def-jna-fn *system-lapack-name* ~(symbol (str "d" fn-name "_"))
       ~description
       nil
       ~@arglist)))


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

(def-lapack-fn potrf
  "Lapack Cholesky decomposition"
  [upload upload-str]
  [N int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [info int-out-value])


(def-lapack-fn potrs
  "Cholesky solve Ax=B system.  A has been cholesky decomposed."
  [upload upload-str]
  [N int-in-value]
  [NRHS int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [B primitive/ensure-ptr-like]
  [ldb int-in-value]
  [info int-out-value])


;;;; LU
(defn trans-str
  [arg]
  (if-let [retval (#{"N" "T" "C"} (str arg))]
    retval
    (throw (ex-info "Argument is not a transpose str"
                    {:arg arg
                     :trans-str-set #{"N" "T" "C"}}))))


(def-lapack-fn getrf
  "LU factorize matrix."
  [M int-in-value]
  [N int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [ipiv primitive/ensure-ptr-like]
  [info int-out-value])


(def-lapack-fn getrs
  "LU solve matrix."
  [trans trans-str]
  [N int-in-value]
  [NRHS int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [ipiv int-ary-in-value]
  [B primitive/ensure-ptr-like]
  [ldb int-in-value]
  [info int-out-value])


;;;; SVD

(def jobu-table
  {:all-columns-U "A"
   :left-singular-U "S"
   :left-singular-A "O"
   :no-singular "N"})

(def jobvt-table
  {:all-rows-VT "A"
   :right-singular-VT "S"
   :right-singular-A "O"
   :no-singular "N"})


(defn to-jobu
  [arg]
  (if-let [retval ((set (vals jobu-table)) (str arg))]
    retval
    (throw (ex-info "Unrecognized jobu argument" {:arg arg}))))


(defn to-jobvt
  [arg]
  (if-let [retval ((set (vals jobvt-table)) (str arg))]
    retval
    (throw (ex-info "Unrecognized jobvt argument" {:arg arg}))))


(def-lapack-fn gesvd
  "*GESVD computes the singular value decomposition (SVD) of a real
 M-by-N matrix A, optionally computing the left and/or right singular
 vectors. The SVD is written

      A = U * SIGMA * transpose(V)

 where SIGMA is an M-by-N matrix which is zero except for its
 min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
 are the singular values of A; they are real and non-negative, and
 are returned in descending order.  The first min(m,n) columns of
 U and V are the left and right singular vectors of A.

 Note that the routine returns V**T, not V."
  [jobu to-jobu]
  [jobvt to-jobvt]
  [M int-in-value]
  [N int-in-value]
  [A primitive/ensure-ptr-like]
  [lda int-in-value]
  [S primitive/ensure-ptr-like] ;;vector, (min M N)
  [U primitive/ensure-ptr-like] ;;matrix, depends on jobu
  [ldu (int-in-value)]
  [VT primitive/ensure-ptr-like] ;;matrix, depends on jobvt
  [ldvt int-in-value]
  [work primitive/ensure-ptr-like] ;;workspace
  [lwork int-in-value] ;;length of work vector
  [info int-out-value] ;;results of method
  )



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
