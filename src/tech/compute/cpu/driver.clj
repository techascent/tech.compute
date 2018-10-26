(ns tech.compute.cpu.driver
  (:require [tech.compute.driver :as drv]
            [tech.datatype :as dtype]
            [tech.datatype.java-primitive :as primitive]
            [tech.datatype.java-unsigned :as unsigned]
            [clojure.core.async :as async]
            [tech.resource :as resource]
            [tech.compute :as compute]
            [tech.compute.registry :as registry]
            [tech.compute.cpu.jna-blas :as jna-blas]
            [tech.datatype.jna :as dtype-jna]
            ;;typed buffer implementation of buffer protocol
            [tech.compute.cpu.typed-buffer]
            ;;typed pointer implementation of buffer protocol
            ;;Also enables javacpp
            [tech.compute.cpu.typed-pointer])
  (:import  [java.nio ByteBuffer ShortBuffer IntBuffer
             LongBuffer FloatBuffer DoubleBuffer]
            [tech.datatype.java_unsigned TypedBuffer]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defrecord CPUDevice [driver-fn device-id error-atom default-stream])
(defrecord CPUStream [device-fn input-chan exit-chan error-atom])
(defrecord CPUDriver [devices error-atom])


(defonce driver-name (registry/current-ns->keyword))


(extend-protocol drv/PDriverProvider
  CPUDriver
  (get-driver [driver] driver)
  CPUDevice
  (get-driver [device] ((get device :driver-fn)))
  CPUStream
  (get-driver [stream] (compute/->driver ((:device-fn stream)))))


(extend-protocol drv/PDeviceProvider
  CPUDriver
  (get-device [driver] (compute/default-device driver))
  CPUDevice
  (get-device [device] device)
  CPUStream
  (get-device [stream] ((:device-fn stream))))


(extend-protocol drv/PStreamProvider
  CPUStream
  (get-stream [stream] stream))


(extend-type CPUStream
  resource/PResource
  (release-resource [impl]
    (when (.input-chan impl)
      (async/close! (.input-chan impl)))))


(defn get-memory-info
  []
  {:free (.freeMemory (Runtime/getRuntime))
   :total (.totalMemory (Runtime/getRuntime))})


(defn cpu-stream
  ([device error-atom]
   (let [^CPUStream retval (->CPUStream (constantly device) (async/chan 16)
                                        (async/chan) error-atom)]
     (async/thread
       (loop [next-val (async/<!! (:input-chan retval))]
         (when next-val
           (try
             (next-val)
             (catch Throwable e
               (reset! error-atom e)))
           (recur (async/<!! (:input-chan retval)))))
       (async/close! (:exit-chan retval)))
     (resource/track retval)))
  ([device] (cpu-stream device (atom nil))))

(declare driver)

(defn main-thread-cpu-stream
  "Create a cpu stream that will execute everything immediately inline.
Use with care; the synchonization primitives will just hang with this stream."
  ^CPUStream []
  (->CPUStream (constantly (compute/default-device (driver))) nil nil nil))


(defn is-main-thread-cpu-stream?
  [^CPUStream stream]
  (not (or (.input-chan stream)
           (.exit-chan stream)
           (.error-atom stream))))


(defn is-thread-cpu-stream?
  [^CPUStream stream]
  (not (is-main-thread-cpu-stream? stream)))


(defn check-stream-error
  [stream]
  (when-let [error-atom (:error-atom stream)]
   (let [error @error-atom]
     (when error
       (compare-and-set! (:error-atom stream) error nil)
       (throw error)))))


(defmacro with-stream-dispatch
  [stream & body]
  `(if (is-thread-cpu-stream? ~stream)
     (do
       (check-stream-error ~stream)
       (let [^CPUStream stream# ~stream]
         (async/>!! (.input-chan stream#)
                    (fn [] ~@body))))
     (do
       ~@body)))


(defrecord CPUEvent [input-chan])


(extend-type CPUStream
  drv/PStream
  (copy-host->device [stream host-buffer host-offset
                      device-buffer device-offset elem-count]
    (with-stream-dispatch stream
      (dtype/copy! host-buffer host-offset device-buffer device-offset elem-count)))
  (copy-device->host [stream device-buffer device-offset host-buffer
                      host-offset elem-count]
    (with-stream-dispatch stream
      (dtype/copy! device-buffer device-offset host-buffer host-offset elem-count)))
  (copy-device->device [stream dev-a dev-a-off dev-b dev-b-off elem-count]
    (with-stream-dispatch stream
      (dtype/copy! dev-a dev-a-off dev-b dev-b-off elem-count)))
  (sync-with-host [stream]
    ;;If main thread cpu stream then we are already syncced
    (when-not (is-main-thread-cpu-stream? stream)
      (let [^CPUEvent event (->CPUEvent (async/chan))]
        (with-stream-dispatch stream
          (async/close! (.input-chan event)))
        (async/<!! (.input-chan event)))))
  (sync-with-stream [src-stream dst-stream]
    (let [^CPUEvent event (->CPUEvent (async/chan))]
      (with-stream-dispatch src-stream
        (async/close! (.input-chan event)))
      (with-stream-dispatch dst-stream
        (async/<!! (.input-chan event))))))


(defn make-cpu-device
  [driver-fn dev-number error-atom]
  (let [retval (->CPUDevice driver-fn dev-number error-atom (atom nil))
        ;;Default cpu stream runs in the main thread of execution
        default-stream (->CPUStream (constantly retval) nil nil nil)]
    (reset! (:default-stream retval) default-stream)
    retval))


(defn- make-typed-thing
  [datatype n-elems]
  (if (jna-blas/has-blas?)
    (dtype-jna/make-typed-pointer datatype n-elems)
    (unsigned/make-typed-buffer datatype n-elems)))



(extend-type CPUDevice
  drv/PDevice

  (memory-info [impl]
    (get-memory-info))

  (supports-create-stream? [device] true)

  (default-stream [device] @(:default-stream device))

  (create-stream [impl]
    (check-stream-error impl)
    (cpu-stream impl (:error-atom impl)))

  (allocate-device-buffer [impl elem-count elem-type options]
    (check-stream-error impl)
    (make-typed-thing elem-type elem-count))

  (device->device-copy-compatible? [src-device dst-device] nil))


(extend-type CPUDriver
  drv/PDriver
  (driver-name [impl]
    driver-name)

  (get-devices [impl]
    @(get impl :devices))

  (allocate-host-buffer [impl elem-count elem-type options]
    (check-stream-error impl)
    (make-typed-thing elem-type elem-count)))


(declare default-cpu-stream)


(def driver
  (memoize
   (fn []
     (let [error-atom (atom nil)
           retval (->CPUDriver (atom nil) error-atom)]
       (reset! (get retval :devices)
               (->> (range 1)
                    (mapv #(make-cpu-device (constantly retval) % error-atom))))
       retval))))


(registry/register-driver (driver))
