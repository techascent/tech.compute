(ns tech.compute.tensor.details
  "Details of tensor operation implementation"
  (:require [tech.compute :as compute]
            [tech.datatype :as dtype]
            [tech.compute.tensor.protocols :as tens-proto]

            [tech.compute.tensor.math :as tm]
            [tech.compute.tensor.defaults :as defaults]
            [tech.compute.tensor.error-checking :as error-checking]
            [tech.compute.tensor.utils :refer [when-not-error
                                               reversev
                                               all-combinations]]
            [tech.compute.tensor.dimensions :as dims]
            [clojure.core.matrix.protocols :as mp]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn datatype->keyword
  [item]
  (cond
    (tens-proto/tensor? item) :tensor
    (number? item) :number))


(defmulti typed-assign!
  "Multimethods for typed assignment."
  (fn
    [dest src options]
    [(datatype->keyword dest)
     (datatype->keyword src)]))


(defmethod typed-assign! [:tensor :number]
  [dest src options]
  (tm/assign-constant! (defaults/infer-stream options dest) dest src))


(defmethod typed-assign! [:tensor :tensor]
  [dest src options]
  (let [dest-ecount (dtype/ecount dest)
        src-ecount (dtype/ecount src)]
    (when-not-error (>= dest-ecount
                        src-ecount)
      "destination element count must be >= src element count"
      {:dest-ecount dest-ecount
       :src-count src-ecount})
    (when-not-error (error-checking/element-counts-commensurate? dest-ecount src-ecount)
      "Src element count must evenly divide dest ecount."
      {:dest-ecount dest-ecount
       :src-ecount src-ecount})
    ;;This should not be here.  If the datatypes match
    ;;then device->device copy is possible assuming they are from
    ;;the same driver.
    (error-checking/ensure-copy-compatible-devices src dest)
    (error-checking/check-partial-alias dest src)
    (if (error-checking/memcpy-semantics? dest src)
      (compute/copy-device->device (tens-proto/tensor->buffer src) 0
                                   (tens-proto/tensor->buffer dest) 0
                                   (dtype/ecount src)
                                   :stream (defaults/infer-stream options dest))
      (do
        (error-checking/ensure-same-device dest src)
        (tm/assign! (defaults/infer-stream options dest) dest src)))))


(defmulti perform-unary-op
  (fn [value op] op))


(defmacro perform-unary-impl
  [op & body]
  `(defmethod perform-unary-op ~op
     [~'value ~'op]
     (let [~'value (double ~'value)]
       ~@body)))


(perform-unary-impl :ceil (Math/ceil value))
(perform-unary-impl :round (Math/round value))
(perform-unary-impl :floor (Math/floor value))
(perform-unary-impl :- (- value))
(perform-unary-impl :tanh (Math/tanh value))
(perform-unary-impl :logistic (/ 1.0
                                 (+ 1.0 (Math/exp (- value)))))
(perform-unary-impl :exp (Math/exp value))
(perform-unary-impl :sqrt (Math/sqrt value))
(perform-unary-impl :noop value)


(defn unary-op!
  "dest[idx] = op(alpha * x)"
  [dest alpha x op options]
  (condp = (datatype->keyword x)
    :number
    (mp/assign! dest (perform-unary-op
                      (* (double (dtype/cast (dtype/get-datatype dest) alpha))
                         (double (dtype/cast (dtype/get-datatype dest) x)))
                      op))
    :tensor
    (if (compute/alias? (tens-proto/tensor->buffer dest)
                        (tens-proto/tensor->buffer x))
      (tm/unary-accum! (defaults/infer-stream options dest) dest alpha op)
      (do
        (error-checking/ensure-datatypes (dtype/get-datatype dest) x)
        (error-checking/ensure-same-device dest x)
        (error-checking/ensure-broadcast-rules dest x)
        (error-checking/check-partial-alias dest x)
        (tm/unary-op! (defaults/infer-stream options dest) dest x alpha op))))
  dest)


(defmulti typed-binary-op
  "Binary operations may contain one or two scalars in various
  positions.  This multimethod disambiguates between those positions."
  (fn [dest alpha x beta y op options]
    [(datatype->keyword x)
     (datatype->keyword y)]))


(defmulti scalar-binary-op
  (fn [lhs rhs op]
    op))


(defmacro impl-scalar-binary-op
  [op & body]
  `(defmethod scalar-binary-op ~op
     [x# y# _#]
     (let [~'x (double x#)
           ~'y (double y#)]
       ~@body)))

(impl-scalar-binary-op :+ (+ x y))
(impl-scalar-binary-op :- (- x y))
(impl-scalar-binary-op :* (* x y))
(impl-scalar-binary-op :/ (/ x y))
(impl-scalar-binary-op :max (Math/max x y))
(impl-scalar-binary-op :min (Math/min x y))


(defmethod typed-binary-op [:number :number]
  [dest alpha x beta y op]
  (mp/assign! dest
              (let [x (* (double alpha) (double x))
                    y (* (double beta) (double y))]
                (scalar-binary-op x y op))))


(defn- binary-op-constant!
  [dest alpha x beta y op reverse-operands? options]
  (error-checking/ensure-broadcast-rules dest x)
  (error-checking/ensure-datatypes (dtype/get-datatype dest) x)
  (let [y (* (double beta) (double y))
        device (compute/->device dest)]
    ;;attempt a strength reduce for a common operation.
    (if (= op :*)
      (unary-op! dest (* (double alpha)
                         (double y))
                 x
                 :noop
                 options)
      (if (compute/alias? (tens-proto/tensor->buffer dest)
                          (tens-proto/tensor->buffer x))
        (tm/binary-accum-constant! (defaults/infer-stream options dest) dest alpha y op
                                   reverse-operands?)
        (do
          (error-checking/check-partial-alias dest x)
          (tm/binary-op-constant! (defaults/infer-stream options dest) dest x alpha y op
                                  reverse-operands?)))))
  dest)


(defmethod typed-binary-op [:tensor :number]
  [dest alpha x beta y op options]
  (binary-op-constant! dest alpha x beta y op false options))


(defmethod typed-binary-op [:number :tensor]
  [dest alpha x beta y op options]
  (binary-op-constant! dest beta y alpha x op true options))


(defmethod typed-binary-op [:tensor :tensor]
  [dest alpha x beta y op options]
  (let [device (compute/->device dest)
        {:keys [max-shape dimensions]} (dims/dimension-seq->max-shape
                                          (tens-proto/tensor->dimensions dest)
                                          (tens-proto/tensor->dimensions x)
                                          (tens-proto/tensor->dimensions y))
        [dest-dims x-dims y-dims] dimensions
        arg-alias? (or (compute/alias? (tens-proto/tensor->buffer dest)
                                       (tens-proto/tensor->buffer x))
                       (compute/alias? (tens-proto/tensor->buffer dest)
                                       (tens-proto/tensor->buffer y)))
        dest-is-max-shape? (= (:shape dest-dims) max-shape)]

    (when-not dest-is-max-shape?
      (when-not arg-alias?
        (throw (ex-info "If destination is a broadcast target then it must be one of the operands"
                        {:destination-dimensions dest-dims
                         :x-dims x-dims
                         :y-dims y-dims
                         :max-shape max-shape}))))
    (if arg-alias?
      (let [x-alias? (compute/alias? (tens-proto/tensor->buffer dest)
                                     (tens-proto/tensor->buffer x))
            [alpha beta y rev-ops?] (if x-alias?
                                      [alpha beta y false]
                                      [beta alpha x true])]
        (error-checking/ensure-broadcast-rules dest y)
        (error-checking/ensure-datatypes (dtype/get-datatype dest) y)
        (tm/binary-accum! (defaults/infer-stream options dest) dest alpha y beta op
                          rev-ops? (not dest-is-max-shape?)))
      (do
        (error-checking/ensure-broadcast-rules dest x y)
        (error-checking/ensure-datatypes (dtype/get-datatype x) y dest)
        (tm/binary-op! (defaults/infer-stream options dest) dest x alpha y beta op))))
  dest)

(defn binary-op!
  "Perform the operation:
dest = alpha * x op beta * y.
x or y may be a scalar, dest must not be.
Datatypes must match."
  [dest alpha x beta y op options]
  (if (= op :-)
    ;;Change minus to plus with negative alpha to minimize situations
    ;;where backends may have a fast path (they only need one for +)
    (typed-binary-op dest alpha x (- (double beta)) y :+ options)
    (typed-binary-op dest alpha x beta y op options))
  dest)


(defn- inline-ternary-op
  [alpha x beta y gamma z op]
  (double
   (condp = op
     :select (if (>= (* (double alpha) (double x))
                     0.0)
               (* (double gamma) (double z))
               (* (double beta) (double y))))))


(defn- order-ternary-args
  [[[x x-dt] [y y-dt] [z z-dt] z-d] alpha beta gamma]
  (let [x-data [x x-dt :x alpha]
        y-data [y y-dt :y beta]
        z-data [z z-dt :z gamma]
        tensor-groups (->> [x-data y-data z-data]
                           (filter #(= :tensor (second %))))
        constant-groups (->> [x-data y-data z-data]
                             (remove #(= :tensor (second %))))]
    {:tensor-pairs (mapv (juxt first #(nth % 3)) tensor-groups)
     :constants (mapv #(* (double (first %))
                          (double (nth % 3))) constant-groups)
     :arg-order (->> (concat (map #(nth % 2) tensor-groups)
                             (map #(nth % 2) constant-groups)))}))


(defn ternary-op!
  "Perform the elementwise operation dest = op( alpha * x, beta * y, gamma * z ) dest
  tensor and must not alias any other arguments.  There is no accumulator version of
  these operations at this time in order to keep kernel permutations low (3 backend
  permutations).

  x, y, z can be constants or tensors.

  operations:
  select: dest = (if (>= x 0) y z)"
  [dest alpha x beta y gamma z op options]
  (let [type-vect (map (juxt identity datatype->keyword) [x y z])
        tensors (->> (filter #(= :tensor (second %)) type-vect)
                     (map first))
        num-tensor-args (count tensors)
        max-ecount (long (apply max 0 (map dtype/ecount tensors)))]
    (if (= 0 num-tensor-args)
      (mp/assign! dest (inline-ternary-op alpha x beta y gamma z op))
      (let [{:keys [tensor-pairs constants arg-order]}
            (order-ternary-args type-vect alpha beta gamma)]
        (apply error-checking/ensure-datatypes (dtype/get-datatype dest) tensors)
        (apply error-checking/ensure-same-device dest tensors)
        (doseq [tens tensors]
          (error-checking/ensure-broadcast-rules dest tens))
        (case num-tensor-args
          3 (tm/ternary-op! (defaults/infer-stream options dest)
                            dest x alpha y
                            beta z gamma op)
          2 (let [[[a-tens a-mul] [b-tens b-mul]] tensor-pairs]
              (tm/ternary-op-constant! (defaults/infer-stream options dest)
                                       dest a-tens a-mul
                                       b-tens b-mul
                                       (first constants) op arg-order))
          1 (let [[[a-tens a-mul]] tensor-pairs]
              (tm/ternary-op-constant-constant! (defaults/infer-stream options dest)
                                                dest a-tens a-mul
                                                (first constants) (second constants)
                                                op arg-order)))))
    dest))


(def unary-reduction-operations
  [:max :min :sum :mean :magnitude-squared :magnitude])


(defn unary-reduce!
  "Vector operations operate across the last dimension and produce 1 result.
output = op((alpha*input))
Output must be a [xyz 1] tensor while input is an [xyz n] tensor;
the reduction will occur across the n axis with the results placed in output.
The leading dimensions of both vectors must match."
  [output alpha input op options]
  (let [output-shape (dtype/shape output)
        input-shape (dtype/shape input)]
    (when-not-error (= (drop-last output-shape)
                       (drop-last input-shape))
      "Output leading dimensions must match input leading dimensions"
      {:output-shape output-shape
       :input-shape input-shape})
    (when-not-error (= 1 (last output-shape))
      "Last dimension of output must be 1"
      {:output-shape output-shape})
    (error-checking/ensure-same-device output input)
    (error-checking/ensure-datatypes (dtype/get-datatype output) input)
    (tm/unary-reduce! (defaults/infer-stream options output) output alpha input op)
    output))
