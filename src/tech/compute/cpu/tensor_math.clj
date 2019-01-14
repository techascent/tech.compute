(ns tech.compute.cpu.tensor-math
  (:require [tech.datatype :as dtype]
            [tech.datatype.java-primitive :as primitive]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.compute.tensor.math :as tm]
            [tech.parallel :as parallel]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.math-util :as cmu]
            [tech.compute.driver :as drv]
            [tech.resource :as resource]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.dimensions :as ct-dims]
            [tech.compute.tensor.utils :as ct-utils]
            [tech.compute.tensor.defaults :as ct-defaults]
            [clojure.core.matrix.stats :as stats]
            [clojure.core.matrix :as m]
            [tech.compute.cpu.driver :as cpu-driver]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     all-datatypes
                     datatype->cast-fn
                     ] :as nio-access]
            [tech.compute.cpu.jna-blas :as jna-blas]
            [tech.datatype.jna :as dtype-jna]
            [tech.compute.cpu.jna-blas :as jna-blas]
            [tech.compute.cpu.jna-lapack :as jna-lapack])
  (:import [tech.compute.cpu.driver CPUStream]
           [java.security SecureRandom]))



(defn- ->buffer
  [tensor] (ct/tensor->buffer tensor))


(defn- ->dimensions
  [tensor] (ct/tensor->dimensions tensor))


(defn assign-constant-map []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.assignment/assign-constant-map))


(defn assign!-map []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.assignment/assign!-map))


(defn unary-op-table []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.unary-op/unary-op-table))


(defn binary-accum-constant-table []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.binary-accum/binary-accum-constant-table))


(defn binary-accum-table []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.binary-accum/binary-accum-table))


(defn binary-op-constant-table []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.binary-op/binary-op-constant-table))


(defn binary-op-table []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.binary-op/binary-op-table))


(defn ternary-op-table []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.ternary-op/ternary-op-table))


(defn unary-reduce-table []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.unary-reduce/unary-reduce-table))


(defn blas-fn-map []
  @(parallel/require-resolve 'tech.compute.cpu.tensor-math.blas/blas-fn-map))


(defn- jna-blas-fn-map
  []
  {[:float32 :gemm] (partial jna-blas/cblas_sgemm :row-major)
   [:float64 :gemm] (partial jna-blas/cblas_dgemm :row-major)})


(defn- jna-lapack-fn-map
  []
  {[:float32 :potrf] jna-lapack/spotrf_
   [:float64 :potrf] jna-lapack/dpotrf_
   [:float32 :potrs] jna-lapack/spotrs_
   [:float64 :potrs] jna-lapack/dpotrs_})


(defn- lapack-upload->fortran
  [upload-kwd]
  (if-let [retval (get
                   {:upper "L"
                    :lower "U"}
                   upload-kwd)]
    retval
    (throw (ex-info "Unrecognized upload command" {:upload-kwd upload-kwd}))))


(extend-type CPUStream
  tm/TensorMath
  (assign-constant! [stream tensor value]
    (cpu-driver/with-stream-dispatch stream
      ;;Use faster, simple fallback if available.
      (if (and (ct/dense? tensor)
               (ct-dims/access-increasing? (ct/tensor->dimensions tensor))
               (= 0.0 (double value))
               (dtype-jna/as-typed-pointer (ct/tensor->buffer tensor)))
        ;;We have memset in this case that will outperform for even very large things.
        (dtype/set-constant! (ct/tensor->buffer tensor) 0 value (ct/ecount tensor))
        ((get (assign-constant-map) (dtype/get-datatype tensor))
         (->buffer tensor) (->dimensions tensor) value (ct/ecount tensor)))))

  (assign! [stream dest src]
    (cpu-driver/with-stream-dispatch stream
      ((get (assign!-map) [(dtype/get-datatype dest) (dtype/get-datatype src)])
       (->buffer dest) (->dimensions dest)
       (->buffer src) (->dimensions src)
       (max (ct/ecount src) (ct/ecount dest)))))

  (unary-accum! [stream dest alpha op]
    (cpu-driver/with-stream-dispatch stream
      ((get-in (unary-op-table) [[(dtype/get-datatype dest) op] :unary-accum!])
       (->buffer dest) (->dimensions dest) alpha (ct/ecount dest))))

  (unary-op! [stream dest x alpha op]
    (cpu-driver/with-stream-dispatch stream
      ((get-in (unary-op-table) [[(dtype/get-datatype dest) op] :unary-op!])
       (->buffer dest) (->dimensions dest) (->buffer x) (->dimensions x) alpha
       (max (ct/ecount dest) (ct/ecount x)))))

  (binary-accum-constant! [stream dest dest-alpha scalar operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get (binary-accum-constant-table) [(dtype/get-datatype dest) operation
                                            reverse-operands?])
       (->buffer dest) (->dimensions dest) dest-alpha
       scalar (ct/ecount dest))))

  (binary-op-constant! [stream dest x x-alpha scalar operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      ((get (binary-op-constant-table) [(dtype/get-datatype dest) operation
                                         reverse-operands?])
       (->buffer dest) (->dimensions dest)
       (->buffer x) (->dimensions x) x-alpha
       scalar (max (ct/ecount dest) (ct/ecount x)))))

  (binary-accum! [stream dest dest-alpha y y-alpha operation
                  reverse-operands? dest-requires-cas?]
    (let [n-elems (max (ct/ecount dest) (ct/ecount y))]
      (if dest-requires-cas?
        (cpu-driver/with-stream-dispatch stream
          ((get (binary-accum-table) [(dtype/get-datatype dest) operation
                                       reverse-operands?])
           (->buffer dest) (->dimensions dest) dest-alpha
           (->buffer y) (->dimensions y) y-alpha
           n-elems))
        ;;If the operation does not require a CAS op then we can use the full
        ;;parallelism of the binary op.  Unfortunately if it does then we have to do a
        ;;lot of things in single-threaded mode.
        (if reverse-operands?
          (tm/binary-op! stream dest y y-alpha dest dest-alpha operation)
          (tm/binary-op! stream dest dest dest-alpha y y-alpha operation)))))

  (binary-op! [stream dest x x-alpha y y-alpha operation]
    (cpu-driver/with-stream-dispatch stream
      ((get (binary-op-table) [(dtype/get-datatype dest) operation])
       (->buffer dest) (->dimensions dest)
       (->buffer x) (->dimensions x) x-alpha
       (->buffer y) (->dimensions y) y-alpha
       (max (ct/ecount x) (ct/ecount y) (ct/ecount dest)))))

  (ternary-op! [stream dest x x-alpha y y-alpha z z-alpha operation]
    (cpu-driver/with-stream-dispatch stream
      ((get-in (ternary-op-table) [(dtype/get-datatype dest) :ternary-op!])
       (->buffer dest) (->dimensions dest)
       (->buffer x) (->dimensions x) x-alpha
       (->buffer y) (->dimensions y) y-alpha
       (->buffer z) (->dimensions z) z-alpha
       (max (ct/ecount x) (ct/ecount y) (ct/ecount z) (ct/ecount dest))
       operation)))

  (ternary-op-constant! [stream dest a a-alpha b b-alpha constant operation arg-order]
    (cpu-driver/with-stream-dispatch stream
      ((get-in (ternary-op-table) [(dtype/get-datatype dest) :ternary-op-constant!])
       (->buffer dest) (->dimensions dest)
       (->buffer a) (->dimensions a) a-alpha
       (->buffer b) (->dimensions b) b-alpha
       constant
       (max (ct/ecount a) (ct/ecount b) (ct/ecount dest))
       operation arg-order)))

  (ternary-op-constant-constant! [stream dest a a-alpha const-1 const-2
                                  operation arg-order]
    (cpu-driver/with-stream-dispatch stream
      ((get-in (ternary-op-table) [(dtype/get-datatype dest)
                                   :ternary-op-constant-constant!])
       (->buffer dest) (->dimensions dest)
       (->buffer a) (->dimensions a) a-alpha
       const-1
       const-2
       (max (ct/ecount a) (ct/ecount dest))
       operation arg-order)))

  (unary-reduce! [stream output input-alpha input op]
    (cpu-driver/with-stream-dispatch stream
      ((get-in (unary-reduce-table) [[(dtype/get-datatype output) op] :unary-reduce!])
      (->buffer output) (->dimensions output)
      input-alpha (->buffer input) (->dimensions input))))

  (gemm! [stream
          C c-colstride
          trans-a? trans-b? alpha
          A a-row-count a-col-count a-colstride
          B b-col-count b-colstride
          beta]
    (if (jna-blas/has-blas?)
      (cpu-driver/with-stream-dispatch stream
        ((get (jna-blas-fn-map) [(dtype/get-datatype C) :gemm])
         trans-a? trans-b? a-row-count b-col-count a-col-count
         alpha (ct/tensor->buffer A) a-colstride
         (ct/tensor->buffer B) b-colstride
         beta (ct/tensor->buffer C) c-colstride))

      ;;Fallback to netlib blas if necessary
      (cpu-driver/with-stream-dispatch stream
        (cmu/col->row-gemm (get-in (blas-fn-map) [(dtype/get-datatype C) :gemm])
                           trans-a? trans-b? a-row-count a-col-count b-col-count
                           alpha (ct/tensor->buffer A) a-colstride
                           (ct/tensor->buffer B) b-colstride
                           beta (ct/tensor->buffer C) c-colstride))))


  (rand! [stream dest {:keys [type] :as distribution}]
    (let [rand-view (item->typed-nio-buffer :float32 (->buffer dest))
          elem-count (ct-dims/ecount (->dimensions dest))
          rand-gen (SecureRandom.)]
      (cond
        (= (:type distribution) :gaussian)
        (let [mean (float (:mean distribution))
              multiplier (Math/sqrt (float (:variance distribution)))]
          (c-for [idx 0 (< idx elem-count) (inc idx)]
                 (let [next-rand (+ (* multiplier (.nextGaussian rand-gen))
                                    mean)]
                   (b-put rand-view idx next-rand))))
        (= (:type distribution) :flat)
        (let [minimum (float (:minimum distribution))
              maximum (float (:maximum distribution))
              range (- maximum minimum)]
         (c-for [idx 0 (< idx elem-count) (inc idx)]
                (b-put rand-view idx (+ minimum
                                         (* (.nextFloat rand-gen)
                                            range)))))
        :else
        (throw (Exception. (str "Unrecognized distribution: " distribution))))))


  tm/LAPACK
  (cholesky-factorize! [stream dest-A upload]
    (cpu-driver/with-stream-dispatch stream
      (if-let [decom-fn (get (jna-lapack-fn-map) [(dtype/get-datatype dest-A) :potrf])]
        (let [[row-count column-count] (ct/shape dest-A)
              fn-retval-ary (int-array 1)
              _ (decom-fn (lapack-upload->fortran upload)
                          (int-array [row-count])
                          (ct/tensor->buffer dest-A)
                          (int-array [column-count])
                          fn-retval-ary)
              fn-retval (aget fn-retval-ary 0)]
          (when-not (= 0 fn-retval)
            (if (< fn-retval 0)
              (throw (ex-info "Internal error, argument incorrect:" {:argument (* -1 fn-retval)}))
              (throw (ex-info "A is not positive definite; detected at column" {:column fn-retval}))))
          dest-A)
        (throw (ex-info "Unable to find decomp function for tensor datatype:"
                        {:datatype (dtype/get-datatype dest-A)})))))

  (cholesky-solve! [stream dest-B upload A]
    (cpu-driver/with-stream-dispatch stream
      (if-let [solve-fn (get (jna-lapack-fn-map) [(dtype/get-datatype dest-B) :potrs])]
        (let [[a-row-count a-col-count] (ct/shape A)
              [b-row-count b-col-count] (ct/shape dest-B)
              fn-retval (int-array 1)
              _ (solve-fn (lapack-upload->fortran upload)
                          (int-array [a-row-count])
                          (int-array [b-row-count])
                          (ct/tensor->buffer A)
                          (int-array [a-col-count])
                          (ct/tensor->buffer dest-B)
                          (int-array [b-col-count])
                          fn-retval)
              fn-retval (aget fn-retval 0)]
          (when (< fn-retval 0)
            (throw (ex-info "Internal error, argument incorrect:" {:argument (* -1 fn-retval)})))
          dest-B)
        (throw (ex-info "Unable to find solve fn for B datatype:"
                        {:datatype (dtype/get-datatype dest-B)}))))))


(defn as-java-array
  [cpu-tensor]
  (drv/sync-with-host (ct-defaults/infer-stream {} cpu-tensor))
  (-> (ct/tensor->buffer cpu-tensor)
      dtype/->array))


(defn buffer->tensor
  "Construct a tensor from a buffer.  It must satisfy either tech.jna/PToPtr or
  tech.datatype.java-unsigned/PToBuffer.  Uses item datatype and shape for tensor."
  [item]
  (if-let [auto-tensor (ct/as-tensor item)]
    auto-tensor
    (if-let [tensor-buffer (or (dtype-jna/as-typed-pointer item)
                               (unsigned/as-typed-buffer item)
                               (unsigned/->typed-buffer item))]
      (ct/construct-tensor (ct-dims/dimensions (ct/shape item))
                           tensor-buffer)
      (throw (ex-info "Unable to construct a tensor or tensor-buffer from item"
                      {:item item})))))
