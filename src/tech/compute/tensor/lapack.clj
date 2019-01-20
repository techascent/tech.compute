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
  returns:
  {:LU LU matrix written into A
   :pivots pivots}"
  [dest-A & {:keys [ipiv row-major?]
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
    (tm/LU-factorize! stream dest-A (ct/ensure-tensor ipiv) row-major?)
    {:LU dest-A
     :pivots ipiv}))


(def trans-cmd-set #{:no-transpose :transpose :conjugate-transpose})


(defn LU-solve!
  "Solve a matrix using an LU factored system"
  [dest-B trans-cmd A pivots & {:keys [row-major?] :as options}]
  (let [dest-B (ct/ensure-tensor dest-B)
        dest-B (if (= 1 (count (ct/shape dest-B)))
                 (if row-major?
                   (ct/in-place-reshape dest-B [(first (ct/shape dest-B)) 1])
                   (ct/in-place-reshape dest-B [1 (first (ct/shape dest-B))])
                   )
                 dest-B)
        A (ct/ensure-tensor A)
        pivots (ct/ensure-tensor pivots)
        stream (defaults/infer-stream options dest-B A pivots)
        ]
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
      (tm/LU-solve! stream dest-B trans-cmd A pivots row-major?)
      dest-B)))


(defn LU-invert-matrix
  "Invert the matrix, leaving it unchanged.  Returns inverted matrix.  row-major? is
  there for testing; the most efficient thing will be to avoid using it in most cases."
  [src-matrix & {:keys [identity row-major?] :as options}]
  (let [inverse (if identity
                  (ct/clone identity)
                  (ct/identity (ct/shape src-matrix)))
        {:keys [LU pivots]} (LU-factorize! (ct/clone src-matrix) :row-major? row-major?)]
    (LU-solve! inverse :no-transpose LU pivots :row-major? row-major?)))


(defn LU-solve-set-of-equations
  "Solve Ax=y for x.
  Should work.  The most efficent pathway will be to not specify row major."
  [A y & {:keys [row-major?]}]
  (let [{:keys [LU pivots]} (LU-factorize! (ct/clone A) :row-major? row-major?)]
    (LU-solve! (ct/clone y) (if row-major?
                              :no-transpose
                              :transpose)
               LU pivots :row-major? row-major?)))


(def jobu-set
  #{:all-columns-U
    :left-singular-U
    :left-singular-A
    :no-singular})

(def jobvt-set
  #{:all-rows-VT
    :right-singular-VT
    :right-singular-A
    :no-singular})


(defn singular-value-decomposition
  "SVD decomposition.  Note that because this is a pure fortran method, A, U, and VT are
  considered transposed implicitly. So jobu set to :all-columns-U really means all-rows-U
  and vice versa."
  [jobu jobvt A s U VT & {:as options}]
  (let [A (ct/ensure-tensor A)
        s (ct/ensure-tensor s)
        U (ct/ensure-tensor U)
        VT (ct/ensure-tensor VT)
        a-datatype (dtype/get-datatype A)
        [a-row-count a-col-count :as a-shape] (ct/shape A)
        u-used? (boolean (#{:all-columns-U
                            :left-singular-U} jobu))
        vt-used? (boolean (#{:all-rows-VT
                             :right-singular-VT} jobvt))
        [u-row-count u-col-count :as u-shape] (if u-used?
                                                (ct/shape U)
                                                [])
        [vt-row-count vt-col-count :as vt-shape] (if vt-used?
                                                   (ct/shape VT)
                                                   [])
        [s-col-count :as s-shape] (ct/shape s)
        N (int a-row-count)
        M (int a-col-count)
        min-m-n (min M N)
        stream (defaults/infer-stream options A s U VT)]
    (error-checking/ensure-datatypes a-datatype s U VT)
    (error-checking/ensure-same-device A s U VT)
    (error-checking/ensure-matrix A)
    (error-checking/ensure-matrix U)
    (error-checking/ensure-matrix VT)
    (when-not-error (= 1 (count s-shape))
      "S must be vector" {:s-shape s-shape})
    (when-not-error (= (int s-col-count)
                       min-m-n)
      "S has invalid shape" {:s-shape s-shape})
    (when-not-error (jobu-set jobu)
      "Invalid jobu" {:jobu jobu
                      :jobu-set jobu-set})

    (when-not-error (or (not u-used?)
                        (= M u-col-count))
      "Invalid U row count" {:u-col-count u-col-count
                             :M M})
    (when u-used?
      (case jobu
        :all-columns-u (when-not-error (= u-shape [M M])
                         "U shape must match be MxM" {:u-shape u-shape
                                                      :M M})
        :left-singular-vectors-U (when-not-error (= u-shape [min-m-n M])
                                   "U shape must be [min-m-n M]" {:u-shape u-shape
                                                                  :min-M-N min-m-n})))

    (when-not-error (jobvt-set jobvt)
      "Invalid jobvt" {:jobvt jobvt
                      :jobvt-set jobvt-set})

    (when-not-error (or (not vt-used?)
                        (= N vt-col-count))
      "Invalid VT col count" {:vt-col-count vt-col-count
                              :N N})

    (when-not-error (jobvt-set jobvt)
      "Invalid jobvt" {:jobvt jobvt
                       :jobvt-set jobvt-set})

    (when vt-used?
      (case jobvt
        :all-rows-vt (when-not-error (= vt-shape [N N])
                       "VT shape must match be NxN" {:vt-shape vt-shape
                                                     :N N})
        :left-singular-VT (when-not-error (= vt-shape [min-m-n M])
                            "VT shape must be [min-m-n N]" {:vt-shape vt-shape
                                                            :min-M-N min-m-n})))

    (when-not-error (not (= #{:left-singular-A :right-singular-A}
                            #{jobu jobvt}))
      "Both jobu and jobvt must not be set to place result in A"
      {:jobu jobu
       :jobvt jobvt})
    (tm/singular-value-decomposition! stream jobu jobvt A s U VT)
    (merge {:s s}
           (when u-used?
             {:U U})
           (when vt-used?
             {:VT VT})
           (when (seq (filter #{:left-singular-A :right-singular-A}
                              [jobu jobvt]))
             {:A A}))))
