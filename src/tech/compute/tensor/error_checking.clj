(ns tech.compute.tensor.error-checking
  (:require [tech.datatype :as dtype]
            [tech.compute.tensor.utils :refer [when-not-error
                                               all-combinations]]
            [tech.compute.tensor.protocols :as tens-proto]
            [tech.compute.tensor.dimensions :as dims]
            [tech.compute :as compute]
            [tech.compute.driver :as compute-drv]))


(defn ensure-datatypes
  "Most tensor functions require all arguments to have same datatype."
  [datatype & args]
  (when-not-error (every? #(= datatype (dtype/get-datatype %)) args)
    "Not all arguments match required datatype"
    {:datatype datatype
     :argument-datatypes (map dtype/get-datatype args)}))


(defn ensure-same-driver
  "Given a set of tensors, ensure they share the same driver."
  [& args]
  (let [driver (compute/->driver (first args))
        wrong-driver (->> (rest args)
                          (remove #(identical? driver (compute/->driver %)))
                          seq)]
    (when-not-error (nil? wrong-driver)
      "Tensor arguments (aside from assignment) must have same driver."
      {})))


(defn same-device?
  [& args]
  (let [first-arg (first args)
        main-device (compute/->device first-arg)]
    (->> (rest args)
         (map #(compute/->device %))
         (every? #(identical? main-device %)))))


(defn ensure-same-device
  "Given a set of tensors, ensure they share the same device.  Only assignment of
  identical types is guaranteed to work across devices."
  [& args]
  (when-not-error (apply same-device? args)
    "Tensor arguments are not all on same device"
    {}))


(defn simple-tensor?
  [tensor]
  (and (tens-proto/dense? tensor)
       (dims/access-increasing? (tens-proto/tensor->dimensions tensor))))


(defn ensure-simple-tensor
  [tensor]
  (when-not-error (simple-tensor? tensor)
    "Tensor must be 'simple' - dense, linearly increasing memory access"
    {:tensor-dimensons (tens-proto/tensor->dimensions tensor)})
  tensor)


(defn memcpy-semantics?
  [dest src]
  (and (= (dtype/ecount dest) (dtype/ecount src))
       (simple-tensor? dest)
       (simple-tensor? src)
       (= (dtype/get-datatype dest)
          (dtype/get-datatype src))))


(defn ensure-copy-compatible-devices
  [src dest]
  (when-not (or (same-device? src dest)
                (compute/device->device-copy-compatible?
                 (compute/->device src)
                 (compute/->device dest)))
    (throw (ex-info "Devices are not copy-compatible"
                    {:src-driver (compute/driver-name (compute/->driver src))
                     :dest-driver (compute/driver-name (compute/->driver dest))}))))


(defn ensure-elementwise-compatible
  "Ensure these two tensors are compatible for an elementwise operation
that rerequires the items to have the same element count."
  [lhs rhs]
  (when-not-error (identical? (compute/->driver lhs)
                              (compute/->driver rhs))
    "Tensor drivers do not match"
    {:lhs lhs
     :rhs rhs})
  (when-not-error (= (dtype/ecount lhs)
                     (dtype/ecount rhs))
    "Tensors must have same ecount for assignment."
    {:lhs-ecount (dtype/ecount lhs)
     :rhs-ecount (dtype/ecount rhs)})
  (when-not-error (= (dtype/get-datatype lhs)
                     (dtype/get-datatype rhs))
    "Tensor datatypes are mismatched"
    {:lhs-datatype (dtype/get-datatype lhs)
     :rhs-datatype (dtype/get-datatype rhs)}))


(defn ensure-assignment-matches
  [dest src]
  ;;In order for marshalling or striding to work we need to ensure
  ;;we are on the same device.  device->device transfers only work with
  ;;a bulk dma transfer and that does not do any marshalling nor does it
  ;;do any indexing.
  (if-not (and (= (dtype/get-datatype dest) (dtype/get-datatype src))
               (tens-proto/dense? dest)
               (tens-proto/dense? src))
    (ensure-same-device dest src)
    (ensure-same-driver dest src)))


(defn check-partial-alias
  [& args]
  (let [partially-overlapping-args
        (->> args
             (map #(tens-proto/tensor->buffer %))
             all-combinations
             (filter #(and (apply same-device? %)
                           (apply compute/partially-alias? %)))
             seq)]
    (when-not-error (nil? partially-overlapping-args)
      "Partially overlapping arguments detected."
      {:args (vec partially-overlapping-args)})))


(defn element-counts-commensurate?
  [^long lhs-ecount ^long rhs-ecount]
  (or (= 0 rhs-ecount)
      (= 0 (rem lhs-ecount rhs-ecount))))


(defn ensure-broadcast-rules
  [& args]
  (let [{:keys [max-shape dimensions]} (->> (map tens-proto/tensor->dimensions args)
                                            (apply dims/dimension-seq->max-shape))
        shape-seq (map dims/shape  dimensions)]
    (when-not-error (every? (fn [shp]
                              (every? #(= 0 (long %))
                                      (map #(rem (long %1) (long %2))
                                           max-shape shp)))
                            shape-seq)
      "Shapes are not broadcast-compatible (dimension counts must be commensurate)"
      {:shapes shape-seq
       :max-shapes max-shape})))


(defn ensure-cudnn-datatype
  [dtype op]
  (when-not-error (or (= :float64 dtype)
                      (= :float32 dtype))
    (format "%s is only defined for float and double tensors" op)
    {:datatype dtype}))


(defn external-library-check!
  [method-name & tensors]
  (apply ensure-datatypes (dtype/get-datatype (first tensors)) (rest tensors))
  (apply ensure-same-device tensors)
  (ensure-cudnn-datatype (dtype/get-datatype (first tensors)) method-name))


(defmacro ensure-matrix
  [item]
  `(when-not-error (= 2 (count (dtype/shape ~item)))
     (format "Argument %s appears to not be a matrix" ~(str item))
     {:shape (dtype/shape ~(str item))}))


(defmacro ensure-vector
  [item]
  `(when-not-error (= 1 (count (dtype/shape ~item)))
     (format "Argument %s appears to not be a vector" ~(str item))
     {:shape (dtype/shape ~(str item))}))


(defn acceptable-tensor-buffer?
  [item]
  (and (satisfies? compute-drv/PDeviceProvider item)
       (satisfies? compute-drv/PDriverProvider item)
       (compute-drv/acceptable-device-buffer? (-> (compute-drv/get-driver item)
                                                  compute/default-device)
                                              item)))
