(ns tech.compute.cpu.jna-blas
  (:require [tech.datatype.jna :as dtype-jna]
            [tech.datatype :as dtype]
            [tech.compute.tensor.error-checking :as error-checking])
  (:import [com.sun.jna Pointer Native Function NativeLibrary]))


(def ^:private system-blas-lib
  (memoize
   (fn []
     (try
       (NativeLibrary/getInstance "blas")
       (catch Throwable e
         (println "Failed to load native blas:" e)
         nil)))))


(def get-blas-fn
  (memoize
   (fn [^String fn-name]
     (when-let [system-blas (system-blas-lib)]
       (.getFunction ^NativeLibrary system-blas fn-name)))))


(defn openblas?
  []
  (boolean
   (get-blas-fn "openblas_get_num_threads")))


(defn openblas-get-num-threads
  []
  (let [^Function blas-fn (get-blas-fn "openblas_get_num_threads")]
    (if blas-fn
      (.invoke blas-fn Integer (object-array 0))
      0)))


(defn openblas-set-num-threads
  [num-threads]
  (let [^Function blas-fn (get-blas-fn "openblas_set_num_threads")]
    (when blas-fn
      (.invoke blas-fn (object-array [(int num-threads)])))))

;; typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
;; typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
;; typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
;; typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
;; typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

(def enums
  {:row-major 101 :column-major 102
   :no-transpose 111 :transpose 112 :conjugate-transpose 113 :conjugate-no-transpose 114
   :upper 121 :lower 122
   :non-unit 131 :unit 132
   :left 141 :right 142})

(defn enum-value
  ^long [enum-name]
  (if-let [retval (get enums enum-name)]
    retval
    (throw (ex-info "Failed to find enum:"
                    {:enum-name enum-name}))))

;; void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order,
;;                  OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
;;                  OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
;;                  OPENBLAS_CONST blasint M,
;;                  OPENBLAS_CONST blasint N,
;;                  OPENBLAS_CONST blasint K,
;; 		 OPENBLAS_CONST float alpha,
;;                  OPENBLAS_CONST float *A,
;;                  OPENBLAS_CONST blasint lda,
;;                  OPENBLAS_CONST float *B,
;;                  OPENBLAS_CONST blasint ldb,
;;                  OPENBLAS_CONST float beta,
;;                  float *C, OPENBLAS_CONST blasint ldc);


(defn has-blas?
  []
  (boolean
   (get-blas-fn "cblas_sgemm")))


(defn- ensure-blas
  []
  (when-not (has-blas?)
    (throw (ex-info "System blas is unavailable." {}))))


(defn- ensure-blas-fn
  [fn-name]
  (if-let [retval (get-blas-fn fn-name)]
    retval
    (throw (ex-info "Blas fn not found" {:fn-name fn-name}))))


(defn bool->blas-transpose
  [trans?]
  (enum-value (if trans? :transpose :no-transpose)))


(defn sgemm
  [order trans-a? trans-b? M N K alpha A lda B ldb beta C ldc]
  (ensure-blas)
  (error-checking/ensure-datatypes :float32 A B C)
  (when-let [blas-fn (ensure-blas-fn "cblas_sgemm")]
    (.invoke blas-fn (object-array
                      [(int (enum-value order))
                       (int (bool->blas-transpose trans-a?))
                       (int (bool->blas-transpose trans-b?))
                       (int M)
                       (int N)
                       (int K)
                       (float alpha)
                       (dtype-jna/->ptr-backing-store A)
                       (int lda)
                       (dtype-jna/->ptr-backing-store B)
                       (int ldb)
                       (float beta)
                       (dtype-jna/->ptr-backing-store C)
                       (int ldc)]))))


(defn dgemm
  [order trans-a? trans-b? M N K alpha A lda B ldb beta C ldc]
  (ensure-blas)
  (error-checking/ensure-datatypes :float64 A B C)
  (when-let [blas-fn (ensure-blas-fn "cblas_dgemm")]
    (.invoke blas-fn (object-array
                      [(int (enum-value order))
                       (int (bool->blas-transpose trans-a?))
                       (int (bool->blas-transpose trans-b?))
                       (int M)
                       (int N)
                       (int K)
                       (double alpha)
                       (dtype-jna/->ptr-backing-store A)
                       (int lda)
                       (dtype-jna/->ptr-backing-store B)
                       (int ldb)
                       (double beta)
                       (dtype-jna/->ptr-backing-store C)
                       (int ldc)]))))
