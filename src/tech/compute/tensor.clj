(ns tech.compute.tensor
  "Functions for dealing with tensors with the compute system"
  (:require [tech.compute.driver :as drv]
            [tech.compute.context :as compute-ctx]
            [tech.v2.datatype :as dtype]
            [tech.v2.tensor.impl :as dtt-impl]
            [tech.v2.tensor.dimensions :as dtt-dims]
            [tech.v2.tensor :as dtt])
  (:import [tech.v2.tensor.impl Tensor]))


(defn new-device-tensor
  ([shape options]
   (let [{:keys [device]} (compute-ctx/options->context options)
         datatype (dtt-impl/default-datatype (:datatype options))
         ecount (long (apply * shape))
         dev-buf (drv/allocate-device-buffer device ecount datatype options)]
     (dtt-impl/construct-tensor dev-buf (dtt-dims/dimensions shape))))
  ([shape]
   (new-device-tensor shape {})))


(defn new-host-tensor
  ([shape options]
   (let [{:keys [driver]} (compute-ctx/options->context options)
         datatype (dtt-impl/default-datatype (:datatype options))
         ecount (long (apply * shape))
         host-buf (drv/allocate-host-buffer driver ecount datatype options)]
     (dtt-impl/construct-tensor host-buf (dtt-dims/dimensions shape))))
  ([shape]
   (new-host-tensor shape {})))


(defn clone-to-device
  "Clone a host tensor to a device.  Tensor must have relatively straighforward
  dimensions (transpose OK but arbitrary reorder or offset not OK) or :force?
  must be specified.
  options:
  :force?  Copy tensor to another buffer if necessary.
  :sync? Sync with stream to ensure copy operation is finished before moving forward."
  ([input-tens options]
   (let [input-tens (dtt/ensure-tensor input-tens)
         dims-incompatible? (not
                             (dtt-impl/simple-dimensions?
                              (dtt-impl/tensor->dimensions input-tens)))
         _ (when (and dims-incompatible?
                     (not (:force? options)))
             (throw (Exception. "Incompatible dimensions and force? not specified")))

         input-tens (if dims-incompatible?
                      (dtt/clone input-tens :container-type :native-buffer)
                      input-tens)
         options (-> (assoc options :datatype (dtype/get-datatype input-tens))
                     (compute-ctx/options->context))
         {:keys [stream]} options
         device-tensor (new-device-tensor (dtype/shape input-tens) options)]
     (drv/copy-host->device stream
                            (dtt/tensor->buffer input-tens) 0
                            (dtt/tensor->buffer device-tensor) 0
                            (dtype/ecount input-tens))
     (when (:sync? options)
       (drv/sync-with-host stream))
     device-tensor))
  ([input-tens]
   (clone-to-device input-tens {})))


(defn ensure-device
  "Ensure a tensor can be used on a device.  Some devices can use CPU tensors."
  ([input-tens options]
   (let [device (or (:device options)
                    (compute-ctx/default-device))]
     (if (and (drv/acceptable-device-buffer? device input-tens)
              (dtt-impl/dims-suitable-for-desc? (dtt-impl/tensor->dimensions
                                                 input-tens)))
       input-tens
       (clone-to-device input-tens))))
  ([input-tens]
   (ensure-device input-tens {})))


(defn clone-to-host
  ([device-tens options]
   (let [dims-incompatible? (not
                             (dtt-impl/simple-dimensions?
                              (dtt-impl/tensor->dimensions device-tens)))
         _ (when (and dims-incompatible?
                     (not (:force? options)))
             (throw (Exception. "Incompatible dimensions and force? not specified")))
         options (assoc options :datatype (dtype/get-datatype device-tens))
         stream (or (:stream options)
                    (drv/default-stream (drv/get-device device-tens)))
         device-tens (if dims-incompatible?
                      (drv/clone-device-tensor stream device-tens options)
                      device-tens)
         host-tensor (new-host-tensor (dtype/shape device-tens) options)]
     (drv/copy-device->host stream
                            (dtt/tensor->buffer device-tens) 0
                            (dtt/tensor->buffer host-tensor) 0
                            (dtype/ecount device-tens))
     (when (:sync? options)
       (drv/sync-with-host stream))
     host-tensor))
  ([device-tens]
   (clone-to-host device-tens {})))


(defn ensure-host
  ([device-tens options]
   (let [driver (drv/get-driver device-tens)]
     (if (drv/acceptable-host-buffer? driver device-tens)
       device-tens
       (clone-to-host device-tens options))))
  ([device-tens]
   (ensure-host device-tens {})))


(extend-type Tensor
  drv/PDriverProvider
  (get-driver [tens]
    (drv/get-driver (dtt/tensor->buffer tens)))
  drv/PDeviceProvider
  (get-device [tens]
    (drv/get-device (dtt/tensor->buffer tens))))
