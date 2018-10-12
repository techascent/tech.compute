(ns tech.compute.cpu.driver
  (:require [tech.compute.driver :as drv]
            [tech.datatype.core :as dtype]
            [tech.datatype.java-primitive :as primitive]
            [tech.datatype.java-unsigned :as unsigned]
            [clojure.core.async :as async]
            [think.resource.core :as resource])
  (:import  [java.nio ByteBuffer ShortBuffer IntBuffer
             LongBuffer FloatBuffer DoubleBuffer]
            [tech.datatype.java_unsigned TypedBuffer]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defrecord CPUDevice [driver-fn device-id error-atom default-stream])
(defrecord CPUStream [device-fn input-chan exit-chan error-atom])
(defrecord CPUDriver [devices error-atom])


(extend-protocol drv/PDriverProvider
  CPUDriver
  (get-driver [driver] driver)
  CPUDevice
  (get-driver [device] ((get device :driver-fn)))
  CPUStream
  (get-driver [stream] (drv/get-driver ((:device-fn stream)))))


(extend-protocol drv/PDeviceProvider
  CPUDriver
  (get-device [driver] (drv/default-device driver))
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
  (->CPUStream (constantly (drv/default-device (driver))) nil nil nil))


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



(extend-type CPUDevice
  drv/PDevice
  (memory-info-impl [impl]
    (get-memory-info))

  (supports-create-stream? [device] true)

  (default-stream [device] @(:default-stream device))

  (create-stream-impl [impl]
    (check-stream-error impl)
    (cpu-stream impl (:error-atom impl)))

  (allocate-device-buffer-impl [impl elem-count elem-type]
    (check-stream-error impl)
    (dtype/make-typed-buffer elem-type elem-count))

  (allocate-rand-buffer-impl [impl elem-count]
    (check-stream-error impl)
    (dtype/make-typed-buffer :float32 elem-count)))


(extend-type CPUDriver
  drv/PDriver
  (get-devices [impl]
    @(get impl :devices))

  (allocate-host-buffer-impl [impl elem-count elem-type options]
    (check-stream-error impl)
    (dtype/make-typed-buffer elem-type elem-count)))


(defn- in-range?
  [^long lhs-off ^long lhs-len ^long rhs-off ^long rhs-len]
  (or (and (>= rhs-off lhs-off)
           (< rhs-off (+ lhs-off lhs-len)))
      (and (>= lhs-off rhs-off)
           (< lhs-off (+ rhs-off rhs-len)))))


(defprotocol PNioPositionLength
  (nio-make-view [item offset len])
  (array-backing-store [item])
  (nio-offset [item])
  (nio-length [item]))


(defmacro to-nio-buf
  [datatype item]
  `(primitive/datatype->buffer-cast-fn ~datatype ~item))


(defmacro implement-pos-length
  [buffer-type datatype]
  `(clojure.core/extend
       ~buffer-type
     PNioPositionLength
     {:nio-make-view (fn [item# offset# len#]
                       ;;We slice the buffer to make it stand-alone
                       (let [buf# (.slice (to-nio-buf ~datatype item#))
                             offset# (long offset#)
                             len# (long len#)]
                         (.position buf# offset#)
                         (.limit buf# (+ offset# len#))
                         buf#))
      :array-backing-store (fn [item#]
                             (let [buf# (to-nio-buf ~datatype item#)]
                               (.array buf#)))
      :nio-offset (fn [item#]
                    (let [buf# (to-nio-buf ~datatype item#)]
                      (.position buf#)))
      :nio-length (fn [item#]
                    (let [buf# (to-nio-buf ~datatype item#)]
                      (- (.limit buf#)
                         (.position buf#))))}))


(defn get-offset
  ^long [item]
  (nio-offset (primitive/->buffer-backing-store item)))


(defn get-length
  ^long [item]
  (nio-length (primitive/->buffer-backing-store item)))


(implement-pos-length ByteBuffer :int8)
(implement-pos-length ShortBuffer :int16)
(implement-pos-length IntBuffer :int32)
(implement-pos-length LongBuffer :int64)
(implement-pos-length FloatBuffer :float32)
(implement-pos-length DoubleBuffer :float64)


(declare default-cpu-stream)


(extend-type TypedBuffer
  drv/PBuffer
  (sub-buffer-impl [buffer offset length]
    (unsigned/->TypedBuffer (nio-make-view (primitive/->buffer-backing-store buffer)
                                           offset length)
                            (dtype/get-datatype buffer)))
  (alias? [lhs rhs]
    (let [lhs-ary (array-backing-store (primitive/->buffer-backing-store lhs))
          rhs-ary (array-backing-store (primitive/->buffer-backing-store rhs))]
      (and (and lhs-ary rhs-ary)
           (identical? lhs-ary rhs-ary)
           (and (= (get-offset lhs)
                   (get-offset rhs))))))
  (partially-alias? [lhs rhs]
    (let [lhs-ary (array-backing-store (primitive/->buffer-backing-store lhs))
          rhs-ary (array-backing-store (primitive/->buffer-backing-store rhs))]
      (and (and lhs-ary rhs-ary)
           (identical? lhs-ary rhs-ary)
           (in-range? (get-offset lhs) (get-length lhs)
                      (get-offset rhs) (get-length rhs)))))
  ;;For uniformity all host/device buffers must implement the resource protocol.
  resource/PResource
  (release-resource [_])
  drv/PDeviceProvider
  (get-device [buffer]
    (drv/get-device (default-cpu-stream)))
  drv/PDriverProvider
  (get-driver [buffer]
    (-> (drv/get-device buffer)
        drv/get-driver)))


(def driver
  (memoize
   (fn []
     (let [error-atom (atom nil)
           retval (->CPUDriver (atom nil) error-atom)]
       (reset! (get retval :devices)
               (->> (range 1)
                    (mapv #(make-cpu-device (constantly retval) % error-atom))))
       retval))))


(def default-cpu-stream
  (memoize
   (fn []
     (-> (driver)
         (drv/default-device)
         (drv/default-stream)))))
