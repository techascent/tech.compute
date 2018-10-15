(ns tech.compute.driver
  "Base set of protocols required to move information from the host to the device as well as
  enable some form of computation on a given device.  There is a cpu implementation provided for
  reference.

  Base datatypes are defined:
   * Driver: Enables enumeration of devices and creation of host buffers.
   * Device: Creates streams and device buffers.
   * Stream: Stream of execution occuring on the device.
   * Event: A synchronization primitive emitted in a stream to notify other
            streams that might be blocking."
  (:require [tech.datatype.core :as dtype]
            [clojure.core.matrix :as m]
            [tech.resource :as resource]))


(defprotocol PDriver
  "A driver is a generic compute abstraction.  Could be a group of threads, could be a machine
  on a network or it could be a CUDA or OpenCL driver.  A stream is a stream of execution
  (analogous to a thread) where subsequent calls are serialized.  All buffers implement a few of
  the datatype interfaces, at least get-datatype and ecount.  Host buffers are expected to
  implement enough of the datatype interfaces to allow a copy operation from generic datatypes
  into them.  This means at least PAccess."
  (get-devices [driver]
    "Get a list of devices accessible to the system.")
  (allocate-host-buffer-impl [driver elem-count elem-type options]
    "Allocate a host buffer.  Transfer from host to device requires data first copied into a
host buffer and then uploaded to a device buffer.
options:
{:usage-type #{:one-time :reusable}
usage-type: Hint to allow implementations to allocate different types of host buffers each
optimized for the desired use case.  Default is one-time."))


(defn allocate-host-buffer
  [driver elem-count elem-type & {:keys [usage-type]
                                  :or {usage-type :one-time}}]
  (when-not (contains? #{:one-time :reusable} usage-type)
    (throw (ex-info "Usage type is not in expected set"
                    {:usage-type usage-type
                     :expected-set #{:one-time :reusable}})))
  (allocate-host-buffer-impl driver elem-count elem-type {:usage-type usage-type}))


(defprotocol PDevice
  (memory-info-impl [device]
    "Get a map of {:free <long> :total <long>} describing the free and total memory in bytes.")
  (supports-create-stream? [device]
    "Does this device support create-stream?")
  (default-stream [device]
    "All devices must have a default stream whether they support create or not.")
  (create-stream-impl [device]
    "Create a stream of execution.  Streams are indepenent threads of execution.  They can be synchronized
with each other and the main thread using events.")
  (allocate-device-buffer-impl [device elem-count elem-type]
    "Allocate a device buffer.  This is the generic unit of data storage used for computation.")
  (allocate-rand-buffer-impl [device elem-count]
    "Allocate a buffer used for rands.  Random number generation in general needs a divisible-by-2 element count
and a floating point buffer (cuda cuRand limitation)"))


(defprotocol PBuffer
  "Interface to create sub-buffers out of larger contiguous buffers."
  (sub-buffer-impl [buffer offset length]
    "Create a sub buffer that shares the backing store with the main buffer.")
  (alias? [lhs-dev-buffer rhs-dev-buffer]
    "Do these two buffers alias each other?  Meaning do they start at the same address
and overlap completely?")
  (partially-alias? [lhs-dev-buffer rhs-dev-buffer]
    "Do these two buffers partially alias each other?  Does some sub-range of their
data overlap?"))


(def ^:dynamic *current-compute-device* nil)


(defmacro unsafe-with-compute-device
  "Unsafe because resources allocated within this block will not necessarily get released on the
  same device.  Building block for building safe device abstractions."
  [device & body]
  `(let [device# ~device]
     (with-bindings {#'*current-compute-device* device#}
       ~@body)))

(defrecord DeviceResources [device res-ctx]
  resource/PResource
  (release-resource [_]
    (unsafe-with-compute-device
     device
     (resource/release-resource-seq res-ctx))))


(defmacro with-compute-device
  "Returns a tuple of retval and tracked device resource buffer"
  [device & body]
  `(let [device# ~device]
     (unsafe-with-compute-device
      device#
      (let [[retval# res-ctx#]
            (resource/return-resource-context
             ~@body)
            dev-resources# (->DeviceResources device# res-ctx#)]
        [retval# (resource/track dev-resources#)]))))


(defn default-device
  [driver]
  (first (get-devices driver)))


(defn current-device
  []
  (when-not *current-compute-device*
    (throw (ex-info "No compute device bound." {})))
  *current-compute-device*)


(defn get-default-stream
  []
  (default-stream (current-device)))


(defn memory-info
  "Get a map of {:free <long> :total <long>} describing the free and total memory in bytes."
  [& {:keys [device]}]
  (memory-info-impl (or device (current-device))))


(defn create-stream
    "Create a stream of execution.  Streams are indepenent threads of execution.  They can be
  synchronized with each other and the main thread using events."
  [& {:keys [device]}]
  (create-stream-impl (or device (current-device))))


(defn allocate-device-buffer
  "Allocate a device buffer.  This is the generic unit of data storage used for computation."
  [elem-count elem-type & {:keys [device]}]
  (allocate-device-buffer-impl (or device (current-device))
                               elem-count elem-type))


(defn allocate-rand-buffer
      "Allocate a buffer used for rands.  Random number generation in general needs a
  divisible-by-2 element count and a floating point buffer (cuda cuRand limitation)"
  [elem-count & {:keys [device]}]
  (allocate-rand-buffer-impl (or device (current-device)) elem-count))


(defprotocol PDriverProvider
  "Get a driver from an object"
  (get-driver [impl]))


(defprotocol PDeviceProvider
  "Get a device from an object."
  (get-device [impl]))


(defprotocol PStream
  "Basic functionality expected of streams.  Streams are an abstraction of a stream of execution
and can be synchonized with the host or with each other using events."
  (copy-host->device [stream host-buffer host-offset
                      device-buffer device-offset elem-count])
  (copy-device->host [stream device-buffer device-offset
                      host-buffer host-offset elem-count])
  (copy-device->device [stream dev-a dev-a-off dev-b dev-b-off elem-count])
  (sync-with-host [stream]
    "Block host until stream's queue is finished executing")
  (sync-with-stream [src-stream dst-stream]
    "Create an event in src-stream's execution queue, then have dst stream wait on
that event.  This allows dst-stream to ensure src-stream has reached a certain point
of execution before continuing.  Both streams must be of the same driver."))


(defprotocol PStreamProvider
  "Get a stream from an object"
  (get-stream [impl]))


(defn sub-buffer
  "Create a view of a buffer which shares the backing store
  with the original.  Offset must be >= 0."
  [device-buffer ^long offset ^long length]
  (let [original-size (dtype/ecount device-buffer)
        new-max-length (- original-size offset)]
    (when-not (<= length new-max-length)
      (throw (ex-info "Sub buffer out of range."
                      {:required new-max-length
                       :current length})))
    (sub-buffer-impl device-buffer offset length)))


(defn host-array->device-buffer
  "Synchronously make a device buffer with these elements in it."
  [stream upload-ary & {:keys [datatype]
                        :or {datatype (dtype/get-datatype upload-ary)}}]
  (let [device (get-device stream)
        driver (get-driver device)
        elem-count (m/ecount upload-ary)
        upload-buffer (allocate-host-buffer driver elem-count datatype
                                            :usage-type :one-time)
        device-buffer (allocate-device-buffer elem-count datatype :device device)]
    (dtype/copy! upload-ary 0 upload-buffer 0 elem-count)
    (copy-host->device stream upload-buffer 0 device-buffer 0 elem-count)
    (sync-with-host stream)
    (resource/release upload-buffer)
    device-buffer))


(defn device-buffer->host-array
  "Synchronously transfer a device buffer to a host array"
  [stream device-buffer]
  (let [device (get-device stream)
        driver (get-driver device)
        elem-count (m/ecount device-buffer)
        datatype (dtype/get-datatype device-buffer)
        download-buffer (allocate-host-buffer driver elem-count datatype
                                              :usage-type :one-time)
        download-ary (dtype/make-array-of-type datatype elem-count)]
    (copy-device->host stream device-buffer 0 download-buffer 0 elem-count)
    (sync-with-host stream)
    (dtype/copy! download-buffer 0 download-ary 0 elem-count)
    (resource/release download-buffer)
    download-ary))
