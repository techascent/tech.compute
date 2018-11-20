(ns tech.compute
  (:require [tech.compute.driver :as drv]
            [tech.compute.registry :as registry]
            [tech.datatype :as dtype]
            [clojure.test :refer :all]
            [tech.resource :as resource]))


(defn driver-names
  "Get the names of the registered drivers."
  []
  (registry/driver-names))


(defn driver
  "Do a registry lookup to find a driver by its name."
  [driver-name]
  (registry/driver driver-name))


(defn ->driver
  "Generically get a driver from a thing"
  [item]
  (drv/get-driver item))


(defn ->device
  "Generically get a device from a thing"
  [item]
  (drv/get-device item))


(defn ->stream
  "Generically get a stream from a thing"
  [item]
  (drv/get-stream item))


(defn driver-name
  [driver]
  (drv/driver-name driver))


(defn get-devices
  [driver]
  (drv/get-devices driver))


(defn default-device
  [driver]
  (first (get-devices driver)))


(defn allocate-host-buffer
  "Allocate a host buffer.  Usage type gives a hint as to the
intended usage of the buffer."
  [driver elem-count elem-type & {:keys [usage-type]
                                  :or {usage-type :one-time}
                                  :as options}]
  (drv/allocate-host-buffer driver elem-count
                            elem-type (assoc options
                                             :usage-type usage-type)))


;; Device API


(defn memory-info
  "Get a map of {:free <long> :total <long>} describing the free and total memory in bytes."
  [device]
  (drv/memory-info device))

(defn supports-create-stream?
  "Does this device support create-stream?"
  [device]
  (drv/supports-create-stream? device))

(defn default-stream
  "All devices must have a default stream whether they support create or not."
  [device]
  (drv/default-stream device))

(defn create-stream
  "Create a stream of execution.  Streams are indepenent threads of execution.  They can
  be synchronized with each other and the main thread using events."
  [device]
  (drv/create-stream device))


(defn allocate-device-buffer
  "Allocate a device buffer.  This is the generic unit of data storage used for
  computation.  No options at this time."
  [device elem-count elem-type & {:as options}]
  (drv/allocate-device-buffer device elem-count elem-type options))


;;Buffer API
(defn sub-buffer
  "Create a sub buffer that shares the backing store with the main buffer."
  ([device-buffer offset length]
   (let [original-size (dtype/ecount device-buffer)
         new-max-length (- original-size offset)]
     (when-not (<= length new-max-length)
       (throw (ex-info "Sub buffer out of range."
                      {:required new-max-length
                       :current length})))
     (drv/sub-buffer device-buffer offset length)))
  ([buffer offset]
   (sub-buffer buffer offset (- (dtype/ecount buffer) offset))))


(defn alias?
  "Do these two buffers alias each other?  Meaning do they start at the same address and
  overlap completely?"
  [lhs-dev-buffer rhs-dev-buffer]
  (drv/alias? lhs-dev-buffer rhs-dev-buffer))


(defn partially-alias?
  "Do these two buffers partially alias each other?  Does some sub-range of their data
  overlap?"
  [lhs-dev-buffer rhs-dev-buffer]
  (drv/partially-alias? lhs-dev-buffer rhs-dev-buffer))


;;Stream API

(defn- check-legal-copy!
  [src-buffer src-offset dst-buffer dst-offset elem-count]
  (let [src-len (- (dtype/ecount src-buffer) (long src-offset))
        dst-len (- (dtype/ecount dst-buffer) (long dst-offset))
        elem-count (long elem-count)]
    (when (> elem-count src-len)
      (throw (ex-info "Copy out of range"
                      {:src-len src-len
                       :elem-count elem-count})))
    (when (> elem-count dst-len)
      (throw (ex-info "Copy out of range"
                      {:dst-len dst-len
                       :elem-count elem-count})))))


(defn- provided-or-default-stream
  [stream device-buffer]
  (or stream (default-stream (->device device-buffer))))


(defn device->device-copy-compatible?
  [src-device dst-device]
  (drv/device->device-copy-compatible? src-device dst-device))


(defn copy-host->device
    "Copy from one device to another.  If no stream is provided then the destination
buffer's device's default stream is used."
  [host-buffer host-offset device-buffer device-offset elem-count & {:keys [stream]}]
  (check-legal-copy! host-buffer host-offset device-buffer device-offset elem-count)
  (let [stream (provided-or-default-stream stream device-buffer)]
    (drv/copy-host->device stream host-buffer host-offset
                           device-buffer device-offset elem-count)))


(defn copy-device->host
    "Copy from one device to another.  If no stream is provided then the source
device buffer's device's default stream is used."
  [device-buffer device-offset host-buffer host-offset elem-count & {:keys [stream]}]
  (check-legal-copy! device-buffer device-offset host-buffer host-offset elem-count)
  (let [stream (provided-or-default-stream stream device-buffer)]
    (drv/copy-device->host stream device-buffer device-offset
                           host-buffer host-offset elem-count)))


(defn copy-device->device
  "Copy from one device to another.  If no stream is provided then the destination
buffer's device's default stream is used."
  [dev-a dev-a-off dev-b dev-b-off elem-count & {:keys [stream]}]
  (check-legal-copy! dev-a dev-a-off dev-b dev-b-off elem-count)
  (let [stream (provided-or-default-stream stream dev-b)]
    (drv/copy-device->device stream dev-a dev-a-off dev-b dev-b-off elem-count)))


(defn sync-with-host
  "Block host until stream's queue is finished executing"
  [stream & [options]]
  (drv/sync-with-host stream))

(defn sync-with-stream
  "Create an event in src-stream's execution queue, then have dst stream wait on that
  event.  This allows dst-stream to ensure src-stream has reached a certain point of
  execution before continuing.  Both streams must be of the same driver."
  [src-stream dst-stream & [options]]
  (drv/sync-with-stream src-stream dst-stream))


(defn copy-host-data->device-buffer
  "Robustly and synchronously make a device buffer with these elements in it.
1.  Make host buffer of correct size.
2.  Make device buffer of correct size.
3.  Copy ary data into host buffer.
4.  Copy host buffer into device buffer.
5.  Wait until operation completes.
6.  Release the host buffer.
7.  Return device buffer."
  [stream upload-ary & {:keys [datatype]
                        :or {datatype (dtype/get-datatype upload-ary)}
                        :as options}]
  (let [device (->device stream)
        driver (->driver device)
        elem-count (dtype/ecount upload-ary)
        device-buffer (allocate-device-buffer device elem-count datatype options)
        upload-buffer (allocate-host-buffer driver elem-count datatype
                                            :usage-type :one-time)]
    (dtype/copy-raw->item! upload-ary upload-buffer 0 options)
    (copy-host->device stream upload-buffer 0 device-buffer 0 elem-count)
    ;;Hold onto host buffer till this completes
    (sync-with-host stream {:gc-root upload-buffer})
    device-buffer))


(defn device-buffer->host-buffer
  "Robustly and synchronously make a device buffer with these elements in it.
1.  Make host buffer of correct size.
2.  Copy device buffer into host buffer.
3.  Wait until operation completes.
4.  Return host buffer."
  [stream device-buffer & {:keys [usage-type]
                           :or {usage-type :one-time}
                           :as options}]
  (let [device (->device stream)
        driver (->driver device)
        elem-count (dtype/ecount device-buffer)
        datatype (dtype/get-datatype device-buffer)
        download-buffer (allocate-host-buffer driver elem-count datatype
                                              (assoc options :usage-type usage-type))]
    (copy-device->host stream device-buffer 0 download-buffer 0 elem-count)
    (sync-with-host stream)
    download-buffer))
