(ns tech.compute.cpu.driver
  (:require [tech.compute.driver :as drv]
            [tech.datatype.core :refer [v-aget v-aset] :as dtype]
            [tech.datatype.base :as dtype-base]
            [tech.datatype.marshal :as marshal]
            [clojure.core.async :as async]
            [think.resource.core :as resource]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [think.parallel.core :as parallel])
  (:import [java.nio ByteBuffer IntBuffer ShortBuffer LongBuffer
            FloatBuffer DoubleBuffer Buffer]
           [com.github.fommil.netlib BLAS]
           [java.util Random]
           [tech.datatype ArrayViewBase
            IntArrayView ByteArrayView ShortArrayView LongArrayView
            FloatArrayView DoubleArrayView]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defrecord CPUDevice [driver-fn device-id error-atom])
(defrecord CPUStream [device input-chan exit-chan error-atom])
(defrecord CPUDriver [devices error-atom])


(extend-protocol drv/PDriverProvider
  CPUDriver
  (get-driver [driver] driver)
  CPUDevice
  (get-driver [device] ((get device :driver-fn)))
  CPUStream
  (get-driver [stream] (drv/get-driver (.device stream))))


(extend-protocol drv/PDeviceProvider
  CPUDriver
  (get-device [driver] (drv/default-device driver))
  CPUDevice
  (get-device [device] device)
  CPUStream
  (get-device [stream] (.device stream)))


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
   (let [^CPUStream retval (->CPUStream device (async/chan 16)
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
  (->CPUStream (drv/default-device (driver)) nil nil nil))


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


(defmacro datatype->view-cast-fn
  [dtype buf]
  (condp = dtype
    :int8 `(marshal/as-byte-array-view ~buf)
    :int16 `(marshal/as-short-array-view ~buf)
    :int32 `(marshal/as-int-array-view ~buf)
    :int64 `(marshal/as-long-array-view ~buf)
    :float32 `(marshal/as-float-array-view ~buf)
    :float64 `(marshal/as-double-array-view ~buf)))


(defmacro datatype->cast-fn
  [dtype val]
  (condp = dtype
    :int8 `(byte ~val)
    :int16 `(short ~val)
    :int32 `(int ~val)
    :int64 `(long ~val)
    :float32 `(float ~val)
    :float64 `(double ~val)))



(defmacro ^:private indexed-copy-impl-macro
  [datatype]
  `(fn [dev-src# dev-src-indexes# src-stride#
        dev-dst# dev-dst-indexes# dst-stride#
        n-elems-per-idx#]
     (let [dev-src# (datatype->view-cast-fn ~datatype dev-src#)
           dev-src-indexes# (datatype->view-cast-fn :int32 dev-src-indexes#)
           src-stride# (long src-stride#)
           dev-dst# (datatype->view-cast-fn ~datatype dev-dst#)
           dev-dst-indexes# (datatype->view-cast-fn :int32 dev-dst-indexes#)
           dst-stride# (long dst-stride#)
           n-elems-per-idx# (long n-elems-per-idx#)
           n-indexes# (.length dev-src-indexes#)]
       (parallel/parallel-for
        vec-idx# n-indexes#
        (let [src-idx# (v-aget dev-src-indexes# vec-idx#)
              dst-idx# (v-aget dev-dst-indexes# vec-idx#)
              src-offset# (* src-idx# src-stride#)
              dst-offset# (* dst-idx# dst-stride#)]
          (c-for
           [elem-idx# 0 (< elem-idx# n-elems-per-idx#) (inc elem-idx#)]
           (v-aset dev-dst# (+ dst-offset# elem-idx#)
                   (v-aget dev-src# (+ src-offset# elem-idx#)))))))))


(defmacro ^:private indexed-copy-iter
  []
  (->> (for [dtype dtype-base/datatypes]
         [dtype `(indexed-copy-impl-macro ~dtype)])
       (into {})))


(def indexed-copy-table
  (indexed-copy-iter))


(extend-type CPUStream
  drv/PStream
  (copy-host->device [stream host-buffer host-offset
                      device-buffer device-offset elem-count]
    (with-stream-dispatch stream
      (dtype/copy! host-buffer host-offset device-buffer device-offset elem-count)))
  (copy-device->host [stream device-buffer device-offset host-buffer host-offset elem-count]
    (with-stream-dispatch stream
      (dtype/copy! device-buffer device-offset host-buffer host-offset elem-count)))
  (copy-device->device [stream dev-a dev-a-off dev-b dev-b-off elem-count]
    (with-stream-dispatch stream
      (dtype/copy! dev-a dev-a-off dev-b dev-b-off elem-count)))
  (memset [stream device-buffer device-offset elem-val elem-count]
    (with-stream-dispatch stream
      (dtype-base/set-constant! device-buffer device-offset elem-val elem-count)))
  (create-event [stream]
    (let [^CPUEvent event (->CPUEvent (async/chan))]
      (with-stream-dispatch stream
        (async/close! (.input-chan event)))
      event))
  (indexed-copy-impl [stream dev-src dev-src-indexes src-stride
                      dev-dst dev-dst-indexes dst-stride
                      n-elems-per-idx]
    (with-stream-dispatch stream
      ((get indexed-copy-table (dtype/get-datatype dev-src))
       dev-src dev-src-indexes src-stride
       dev-dst dev-dst-indexes dst-stride
       n-elems-per-idx)))
  (sync-event [stream event]
    (when-not (is-main-thread-cpu-stream? stream)
      (with-stream-dispatch stream
        (drv/wait-for-event event)))))

(extend-type CPUEvent
  drv/PEvent
  (wait-for-event [event]
    (async/<!! (.input-chan event)))
  resource/PResource
  (release-resource [event]))

(defn driver
  [& {:keys [num-devices]
      :or {num-devices 1}}]
  (let [error-atom (atom nil)
        retval (->CPUDriver (atom nil) error-atom)]
    (reset! (get retval :devices)
            (->> (range num-devices)
                 (mapv #(-> (->CPUDevice (constantly retval) % error-atom)))))
    retval))

(extend-type CPUDevice
  drv/PDevice
  (memory-info-impl [impl]
    (get-memory-info))

  (create-stream-impl [impl]
    (check-stream-error impl)
    (cpu-stream impl (:error-atom impl)))

  (allocate-device-buffer-impl [impl elem-count elem-type]
    (check-stream-error impl)
    (dtype/make-view elem-type elem-count))

  (allocate-rand-buffer-impl [impl elem-count]
    (check-stream-error impl)
    (dtype/make-view :float32 elem-count)))


(extend-type CPUDriver
  drv/PDriver
  (get-devices [impl]
    @(get impl :devices))

  (allocate-host-buffer-impl [impl elem-count elem-type options]
    (check-stream-error impl)
    (dtype/make-view elem-type elem-count)))


(defn- in-range?
  [^long lhs-off ^long lhs-len ^long rhs-off ^long rhs-len]
  (or (and (>= rhs-off lhs-off)
           (< rhs-off (+ lhs-off lhs-len)))
      (and (>= lhs-off rhs-off)
           (< lhs-off (+ rhs-off rhs-len)))))


(defmacro array-view-pbuffer-impl
  [view-type cast-fn copy-fn dtype-fn]
  `(extend-type ~view-type
     drv/PBuffer
     (sub-buffer-impl [buffer# offset# length#]
       (dtype/->view buffer# offset# length#))
     (alias? [lhs-dev-buffer# rhs-dev-buffer#]
       (when (identical? (type lhs-dev-buffer#)
                         (type rhs-dev-buffer#))
         (let [lhs-buf# (~cast-fn lhs-dev-buffer#)
               rhs-buf# (~cast-fn rhs-dev-buffer#)]
           (and (identical? (.data lhs-buf#)
                            (.data rhs-buf#))
                (= (.offset lhs-buf#)
                   (.offset rhs-buf#))))))
     (partially-alias? [lhs-dev-buffer# rhs-dev-buffer#]
       (when (identical? (type lhs-dev-buffer#)
                         (type rhs-dev-buffer#))
         (let [lhs-buf# (~cast-fn lhs-dev-buffer#)
               rhs-buf# (~cast-fn rhs-dev-buffer#)]
           (and (identical? (.data lhs-buf#)
                            (.data rhs-buf#))
                (in-range? (.offset lhs-buf#) (.length lhs-buf#)
                           (.offset rhs-buf#) (.length rhs-buf#))))))))

(defmacro array-view-iterator
  [inner-macro & args]
  `[(~inner-macro ByteArrayView marshal/as-byte-array-view 'copy-to-byte-array byte ~@args)
    (~inner-macro ShortArrayView marshal/as-short-array-view 'copy-to-short-array short ~@args)
    (~inner-macro IntArrayView marshal/as-int-array-view 'copy-to-int-array int ~@args)
    (~inner-macro LongArrayView marshal/as-long-array-view 'copy-to-long-array long ~@args)
    (~inner-macro FloatArrayView marshal/as-float-array-view 'copy-to-float-array float ~@args)
    (~inner-macro DoubleArrayView marshal/as-double-array-view 'copy-to-double-array double ~@args)])


(array-view-iterator array-view-pbuffer-impl)


(extend-type ArrayViewBase
  resource/PResource
  (release-resource [_]))


(def default-cpu-stream
  (memoize
   (fn []
     (main-thread-cpu-stream))))
