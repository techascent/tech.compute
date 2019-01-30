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
           [java.security SecureRandom]
           [com.sun.jna Pointer]
           [tech.compute.cpu UnaryOp BinaryOp UnaryReduce]))



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


(def ^:dynamic custom-op-table (atom {}))


(defn add-unary-op!
  [keywd unary-op]
  (when-not (instance? UnaryOp unary-op)
    (throw (ex-info "Operand is not a tech.compute.cpu.UnaryOp"
                    {:operand unary-op})))
  (swap! custom-op-table assoc [keywd] {:type :unary
                                        :operand unary-op})
  keywd)


(defn add-binary-op!
  [keywd binary-op]
  (when-not (instance? BinaryOp binary-op)
    (throw (ex-info "Operand is not a tech.compute.cpu.BinaryOp"
                    {:operand binary-op})))
  (swap! custom-op-table assoc [keywd] {:type :binary
                                        :operand binary-op})
  keywd)


(defn add-unary-reduce!
  [keywd unary-reduce-op]
  (when-not (instance? UnaryReduce unary-reduce-op)
    (throw (ex-info "Operand is not a tech.compute.cpu.BinaryOp"
                    {:operand unary-reduce-op})))
  (swap! custom-op-table assoc [keywd] {:type :unary-reduce
                                        :operand unary-reduce-op})
  keywd)


(defn- jna-blas-fn-map
  []
  {[:float32 :gemm] (partial jna-blas/cblas_sgemm :row-major)
   [:float64 :gemm] (partial jna-blas/cblas_dgemm :row-major)})


(defn- jna-lapack-fn-map
  []
  {[:float32 :potrf] jna-lapack/spotrf_
   [:float64 :potrf] jna-lapack/dpotrf_
   [:float32 :potrs] jna-lapack/spotrs_
   [:float64 :potrs] jna-lapack/dpotrs_

   [:float32 :getrf] jna-lapack/sgetrf_
   [:float64 :getrf] jna-lapack/dgetrf_
   [:float32 :getrs] jna-lapack/sgetrs_
   [:float64 :getrs] jna-lapack/dgetrs_

   [:float32 :gesvd] jna-lapack/sgesvd_
   [:float64 :gesvd] jna-lapack/dgesvd_})


(defn- lapack-upload->fortran
  [upload-kwd]
  (if-let [retval (get
                   {:upper "L"
                    :lower "U"}
                   upload-kwd)]
    retval
    (throw (ex-info "Unrecognized upload command" {:upload-kwd upload-kwd}))))


(defn- lapack-trans->fortran
  [trans-kwd]
  (if-let [retval (get {:no-transpose "N"
                        :transpose "T"
                        :conjugate-transpose "C"}
                       trans-kwd)]
    retval
    (throw (ex-info "Failed to get correct transpose cmd" {:trans-kwd trans-kwd}))))


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


(defn- jobu->fortran
  [jobu]
  (if-let [retval (get jobu-table jobu)]
    retval
    (throw (ex-info "Unrecognized jobu" {:jobu jobu}))))


(defn- jobvt->fortran
  [jobvt]
  (if-let [retval (get jobvt-table jobvt)]
    retval
    (throw (ex-info "Unrecognized jobu" {:jobvt jobvt}))))


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
      (if-let [built-in (get-in (unary-op-table) [[(dtype/get-datatype dest) op] :unary-accum!])]
        (built-in
         (->buffer dest) (->dimensions dest) alpha (ct/ecount dest))
        (if-let [{:keys [type operand]} (get @custom-op-table op)]
          (do
            (when-not (= type :unary)
              (throw (ex-info "Custom operand is not a unary op"
                              {:op-type type})))
            (let [built-in (get-in (unary-op-table) [[(dtype/get-datatype dest) :custom] :unary-accum!])]
              (built-in
               (->buffer dest) (->dimensions dest) alpha (ct/ecount dest) operand)))
          (throw (ex-info "Failed to find operand" {:operand op}))))))

  (unary-op! [stream dest x alpha op]
    (cpu-driver/with-stream-dispatch stream
      (if-let [built-in (get-in (unary-op-table) [[(dtype/get-datatype dest) op] :unary-op!])]
        (built-in
         (->buffer dest) (->dimensions dest) (->buffer x) (->dimensions x) alpha
         (max (ct/ecount dest) (ct/ecount x)))
        (if-let [{:keys [type operand]} (get @custom-op-table op)]
          (do
            (when-not (= type :unary)
              (throw (ex-info "Custom operand is not a unary op"
                              {:op-type type})))
            (let [built-in (get-in (unary-op-table) [[(dtype/get-datatype dest) :custom] :unary-op!])]
              (built-in
               (->buffer dest) (->dimensions dest) alpha (ct/ecount dest) operand)))
          (throw (ex-info "Failed to find operand" {:operand op}))))))

  (binary-accum-constant! [stream dest dest-alpha scalar operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      (if-let [built-in (get (binary-accum-constant-table) [(dtype/get-datatype dest) operation
                                                            reverse-operands?])]
        (built-in
         (->buffer dest) (->dimensions dest) dest-alpha
         scalar (ct/ecount dest))
        (if-let [{:keys [type operand]} (get @custom-op-table operation)]
          (do
            (when-not (= type :binary)
              (throw (ex-info "Custom operand is not a binary op"
                              {:op-type type})))
            (let [built-in (get (binary-accum-constant-table) [(dtype/get-datatype dest) :custom])]
              (built-in (->buffer dest) (->dimensions dest) dest-alpha
                        scalar (ct/ecount dest)
                        operand reverse-operands?)))
          (throw (ex-info "Failed to find operand" {:operand operation}))))))

  (binary-op-constant! [stream dest x x-alpha scalar operation reverse-operands?]
    (cpu-driver/with-stream-dispatch stream
      (if-let [built-in (get (binary-op-constant-table) [(dtype/get-datatype dest) operation
                                                         reverse-operands?])]
        (built-in
         (->buffer dest) (->dimensions dest)
         (->buffer x) (->dimensions x) x-alpha
         scalar (max (ct/ecount dest) (ct/ecount x)))
        (if-let [{:keys [type operand]} (get @custom-op-table operation)]
          (do
            (when-not (= type :binary)
              (throw (ex-info "Custom operand is not a unary op"
                              {:op-type type})))
            (let [built-in (get (binary-op-constant-table) [(dtype/get-datatype dest) :custom])]
              (built-in
               (->buffer dest) (->dimensions dest)
               (->buffer x) (->dimensions x) x-alpha
               scalar (max (ct/ecount dest) (ct/ecount x))
               operand reverse-operands?)))
          (throw (ex-info "Failed to find operand" {:operand operation}))))))

  (binary-accum! [stream dest dest-alpha y y-alpha operation
                  reverse-operands? dest-requires-cas?]
    (let [n-elems (max (ct/ecount dest) (ct/ecount y))]
      (if dest-requires-cas?
        (cpu-driver/with-stream-dispatch stream
          (if-let [built-in (get (binary-accum-table) [(dtype/get-datatype dest) operation
                                                       reverse-operands?])]
            (built-in
             (->buffer dest) (->dimensions dest) dest-alpha
             (->buffer y) (->dimensions y) y-alpha
             n-elems)
            (if-let [{:keys [type operand]} (get @custom-op-table operation)]
              (do
                (when-not (= type :binary)
                  (throw (ex-info "Custom operand is not a unary op"
                                  {:op-type type})))
                (let [built-in (get (binary-accum-table) [(dtype/get-datatype dest) :custom])]
                  (built-in
                   (->buffer dest) (->dimensions dest) dest-alpha
                   (->buffer y) (->dimensions y) y-alpha
                   n-elems operand reverse-operands?)))
              (throw (ex-info "Failed to find operand" {:operand operation})))))
        ;;If the operation does not require a CAS op then we can use the full
        ;;parallelism of the binary op.  Unfortunately if it does then we have to do a
        ;;lot of things in single-threaded mode.
        (if reverse-operands?
          (tm/binary-op! stream dest y y-alpha dest dest-alpha operation)
          (tm/binary-op! stream dest dest dest-alpha y y-alpha operation)))))

  (binary-op! [stream dest x x-alpha y y-alpha operation]
    (cpu-driver/with-stream-dispatch stream
      (if-let [built-in (get (binary-op-table) [(dtype/get-datatype dest) operation])]
        (built-in
         (->buffer dest) (->dimensions dest)
         (->buffer x) (->dimensions x) x-alpha
         (->buffer y) (->dimensions y) y-alpha
         (max (ct/ecount x) (ct/ecount y) (ct/ecount dest)))
        (if-let [{:keys [type operand]} (get @custom-op-table operation)]
          (do
            (when-not (= type :binary)
              (throw (ex-info "Custom operand is not a unary op"
                              {:op-type type})))
            (let [built-in (get (binary-op-table) [(dtype/get-datatype dest) :custom])]
              (built-in
               (->buffer dest) (->dimensions dest)
               (->buffer x) (->dimensions x) x-alpha
               (->buffer y) (->dimensions y) y-alpha
               (max (ct/ecount x) (ct/ecount y) (ct/ecount dest))
               operand)))
          (throw (ex-info "Failed to find operand" {:operand operation}))))))


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
    (if-let [decom-fn (get (jna-lapack-fn-map) [(dtype/get-datatype dest-A) :potrf])]
      (cpu-driver/with-stream-dispatch stream
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
          dest-A))
      (throw (ex-info "Unable to find decomp function for tensor datatype:"
                      {:datatype (dtype/get-datatype dest-A)}))))

  (cholesky-solve! [stream dest-B upload A]
    (if-let [solve-fn (get (jna-lapack-fn-map) [(dtype/get-datatype dest-B) :potrs])]
      (cpu-driver/with-stream-dispatch stream
        (let [[a-row-count a-col-count] (ct/shape A)
              [b-row-count b-col-count] (ct/shape dest-B)
              fn-retval (int-array 1)
              _ (solve-fn (lapack-upload->fortran upload)
                          a-row-count b-row-count
                          (ct/tensor->buffer A)
                          a-col-count
                          (ct/tensor->buffer dest-B)
                          b-col-count
                          fn-retval)
              fn-retval (aget fn-retval 0)]
          (when (< fn-retval 0)
            (throw (ex-info "Internal error, argument incorrect:" {:argument (* -1 fn-retval)})))
          dest-B))
      (throw (ex-info "Unable to find solve fn for B datatype:"
                      {:datatype (dtype/get-datatype dest-B)}))))

  (LU-factorize! [stream dest-A dest-ipiv row-major?]
    (if-let [factor-fn (get (jna-lapack-fn-map) [(dtype/get-datatype dest-A) :getrf])]
      (cpu-driver/with-stream-dispatch stream
        ;;We have to make class to do transpose or clone in this thread.
        (ct-defaults/with-stream (cpu-driver/main-thread-cpu-stream)
          (let [orig-A dest-A
                dest-A (if row-major?
                         (-> dest-A
                             (ct/transpose [1 0])
                             (ct/clone))
                         dest-A)
                [a-row-count a-col-count] (ct/shape dest-A)
                [n-ipiv-rows] (ct/shape dest-ipiv)
                fn-retval (int-array 1)
                _ (factor-fn a-col-count a-row-count (ct/tensor->buffer dest-A)
                             a-col-count (ct/tensor->buffer dest-ipiv)
                             fn-retval)
                retval (aget fn-retval 0)]
            (cond
              (= retval 0) {:LU (if row-major?
                                  (ct/assign! orig-A (ct/transpose dest-A [1 0]))
                                  dest-A)
                            :pivots dest-ipiv}
              (< retval 0) (throw (ex-info "I-th argument error:" {:i retval}))
              (> retval 0) (throw (ex-info "U is singular" {:column retval}))))))
      (throw (ex-info "Unable to find decomp function for tensor datatype:"
                      {:datatype (dtype/get-datatype dest-A)}))))


  (LU-solve! [stream dest-B trans A ipiv row-major?]
    (if-let [solve-fn (get (jna-lapack-fn-map) [(dtype/get-datatype dest-B) :getrs])]
      (cpu-driver/with-stream-dispatch stream
        (ct-defaults/with-stream (cpu-driver/main-thread-cpu-stream)
          (let [
                trans-cmd (lapack-trans->fortran trans)

                A (if row-major?
                    (-> (ct/transpose A [1 0])
                        (ct/clone))
                    A)
                orig-B dest-B
                dest-B (if row-major?
                         (-> (ct/transpose dest-B [1 0])
                             (ct/clone))
                         dest-B)
                [a-row-count a-col-count] (ct/shape A)
                [b-row-count b-col-count] (ct/shape dest-B)
                fn-retval (int-array 1)
                _ (solve-fn trans-cmd
                            a-row-count b-row-count
                            (ct/tensor->buffer A)
                            a-col-count
                            (ct/tensor->buffer ipiv)
                            (ct/tensor->buffer dest-B)
                            b-col-count
                            fn-retval)
                retval (aget fn-retval 0)]
            (cond
              (= 0 retval) (if row-major?
                             (ct/assign! orig-B (ct/transpose dest-B [1 0]))
                             dest-B)
              (< retval 0) (throw (ex-info "ith argument had error" {:i retval}))))))
      (throw (ex-info "Failed to find lu solve for datatype" {:datatype (dtype/get-datatype dest-B)}))))


  (singular-value-decomposition! [stream jobu jobvt A s U VT]
    (if-let [lapack-fn (get (jna-lapack-fn-map) [(dtype/get-datatype A) :gesvd])]
      (cpu-driver/with-stream-dispatch stream
        (let [jobu (jobu->fortran jobu)
              jobvt (jobvt->fortran jobvt)
              retval-data (int-array [1])
              work (dtype/make-array-of-type (dtype/get-datatype A) [1])
              [N M] (ct/shape A)
              [u-row-count u-col-count] (if U
                                          (ct/shape U)
                                          [0 0])
              [vt-row-count vt-col-count] (if VT
                                            (ct/shape VT)
                                            [0 0])
              A (ct/tensor->buffer A)
              s (ct/tensor->buffer s)
              U (if U
                  (ct/tensor->buffer U)
                  (Pointer. 0))
              VT (if VT
                   (ct/tensor->buffer VT)
                   (Pointer. 0))
              lapack-closure #(lapack-fn jobu jobvt M N
                                         A M
                                         s
                                         U
                                         u-col-count
                                         VT
                                         vt-col-count
                                         %1
                                         %2
                                         retval-data)
              _ (lapack-closure work -1)
              _ (when-not (= 0 (aget retval-data 0))
                  (throw (ex-info "Failure in work query" {:retval (aget retval-data 0)})))
              lwork (long (dtype/get-value work 0))
              work (dtype/make-array-of-type (dtype/get-datatype A) lwork)
              _ (lapack-closure work lwork)
              retval (aget retval-data 0)]
          (cond
            (= retval 0) {:A A :s s :U U :VT VT}
            (< retval 0) (throw (ex-info "i-th argument had an illegal value" {:ith retval}))
            (> retval 0) (throw (ex-info "DBSQR did not converge" {:unconverged-superdiagonals retval})))))
      (throw (ex-info "Failed to find SVD for datatype" {:datatype (dtype/get-datatype A)})))))


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
