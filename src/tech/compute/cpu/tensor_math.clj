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
            [tech.datatype.jna :as dtype-jna])
  (:import [tech.compute.cpu.driver CPUStream]
           [java.security SecureRandom]))



(defn- ->buffer
  [tensor] (ct/tensor->buffer tensor))


(defn- ->dimensions
  [tensor] (ct/tensor->dimensions tensor))


(def ^:private lock-object (Object.))


(defmacro ^:private threadsafe-constant
  [& body]
  `(memoize
    (fn []
      ;;There is no guarantee that require or resolve, called in many threads at once
      ;;return what you think they might.  So we ensure exactly one thread at a time in
      ;;this code.
      (locking lock-object
        ~@body))))


(def assign-constant-map
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.assignment])
   @(resolve 'tech.compute.cpu.tensor-math.assignment/assign-constant-map)))


(def assign!-map
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.assignment])
   @(resolve 'tech.compute.cpu.tensor-math.assignment/assign!-map)))


(def unary-op-table
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.unary-op])
   @(resolve 'tech.compute.cpu.tensor-math.unary-op/unary-op-table)))


(def binary-accum-constant-table
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.binary-accum])
   @(resolve 'tech.compute.cpu.tensor-math.binary-accum/binary-accum-constant-table)))


(def binary-accum-table
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.binary-accum])
   @(resolve 'tech.compute.cpu.tensor-math.binary-accum/binary-accum-table)))


(def binary-op-constant-table
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.binary-op])
   @(resolve 'tech.compute.cpu.tensor-math.binary-op/binary-op-constant-table)))


(def binary-op-table
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.binary-op])
   @(resolve 'tech.compute.cpu.tensor-math.binary-op/binary-op-table)))


(def ternary-op-table
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.ternary-op])
   @(resolve 'tech.compute.cpu.tensor-math.ternary-op/ternary-op-table)))


(def unary-reduce-table
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.unary-reduce])
   @(resolve 'tech.compute.cpu.tensor-math.unary-reduce/unary-reduce-table)))


(def blas-fn-map
  (threadsafe-constant
   (require '[tech.compute.cpu.tensor-math.blas])
   @(resolve 'tech.compute.cpu.tensor-math.blas/blas-fn-map)))



(defn- jna-blas-fn-map
  []
  {[:float32 :gemm] (partial jna-blas/sgemm :row-major)
   [:float64 :gemm] (partial jna-blas/dgemm :row-major)})


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
    (if (every? dtype-jna/typed-pointer? (->> [A B C]
                                              (map ->buffer)))
      (cpu-driver/with-stream-dispatch stream
        ((get (jna-blas-fn-map) [(dtype/get-datatype C) :gemm])
         trans-a? trans-b? a-row-count b-col-count a-col-count
         alpha (ct/tensor->buffer A) a-colstride
         (ct/tensor->buffer B) b-colstride
         beta (ct/tensor->buffer C) c-colstride))


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
        (throw (Exception. (str "Unrecognized distribution: " distribution)))))))



(defn as-java-array
  [cpu-tensor]
  (drv/sync-with-host (ct-defaults/infer-stream {} cpu-tensor))
  (let [dev-buffer (ct/tensor->buffer cpu-tensor)]
    (dtype/->array dev-buffer)))


(defn buffer->tensor
  "Construct a tensor from a buffer.  It must satisfy either
tech.datatype.jna/PToPtr or tech.datatype.java-unsigned/PToBuffer.
Uses item datatype and shape for tensor."
  [item]
  (let [tensor-buffer (if (satisfies? dtype-jna/PToPtr item)
                        (or (dtype-jna/as-typed-pointer item)
                            (dtype-jna/->typed-pointer item))
                        (or (unsigned/as-typed-buffer item)
                            (unsigned/->typed-buffer item)))]
    (ct/construct-tensor (ct-dims/dimensions (ct/shape item))
                         tensor-buffer)))


(defn as-tensor
  [item]
  (if (ct/tensor? item)
    item
    (buffer->tensor item)))
