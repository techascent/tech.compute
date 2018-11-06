(ns tech.compute.tensor.defaults
  (:require [tech.compute :as compute]
            [tech.datatype :as dtype]
            [tech.compute.tensor.protocols :as tens-proto]
            [tech.compute.registry :as registry]))


;;In order to take arbitrary clojure datastructures and create a tensor
;;we need to have a couple global variables bound: Stream and datatype.
;;In general all tensor operations rely on a stream.
;;Stream lookup goes in this order:
;;1.  Stream provided directly through arguments to function.
;;2.  Global stream if set.
;;3.  Default stream of the destnation tensor's buffer's device


(def ^:dynamic *stream* nil)


(defmacro with-stream
  [stream & body]
  `(with-bindings {#'*stream* ~stream}
     ~@body))


(defn unsafe-set-stream!
  "Not-threadsafe set global-stream"
  [stream-fn]
  (alter-var-root #'*stream* stream-fn))


(defn infer-stream
  [options & tensor-args]
  (if-let [retval (or (:stream options)
                      *stream*
                      (->> tensor-args
                           (filter tens-proto/tensor?)
                           (map (comp compute/default-stream
                                      compute/->device))
                           first)
                      ;;Lastly, use the currently bound cpu driver
                      (when-let [driver-name (registry/cpu-driver-name)]
                        (-> (registry/driver driver-name)
                            compute/default-device
                            compute/default-stream)))]
    retval
    (throw (ex-info "Stream is unset and no tensor arguments found."
                    {}))))

;;Similar to stream, the engine will set this variable and clients should not set
;;the variable themselves.
(def ^:dynamic *datatype* :float64)

(defmacro with-datatype
  [dtype & body]
  `(with-bindings {#'*datatype* ~dtype}
     ~@body))


(defn datatype
  [dtype-or-nil]
  (or dtype-or-nil *datatype*))


(defmacro tensor-context
  [stream datatype & body]
  `(with-stream ~stream
     (with-datatype ~datatype
       ~@body)))


(defmacro tensor-driver-context
  [driver datatype & body]
  `(with-stream (-> (compute/default-device ~driver)
                    compute/default-stream)
     (with-datatype ~datatype
       ~@body)))
