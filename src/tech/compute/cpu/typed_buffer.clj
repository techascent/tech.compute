(ns tech.compute.cpu.typed-buffer
  (:require [tech.datatype :as dtype]
            [tech.datatype.java-primitive :as primitive]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.compute.driver :as drv]
            [tech.compute :as compute]
            [tech.compute.registry :as registry]
            [tech.compute.cpu.utils :as cpu-utils])
    (:import  [java.nio ByteBuffer ShortBuffer IntBuffer
               LongBuffer FloatBuffer DoubleBuffer]
              [tech.datatype.java_unsigned TypedBuffer]))



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
                             (dtype/->array item#))
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


(extend-type TypedBuffer
  drv/PBuffer
  (sub-buffer [buffer offset length]
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
           (cpu-utils/in-range? (get-offset lhs) (get-length lhs)
                                (get-offset rhs) (get-length rhs)))))
  drv/PDeviceProvider
  (get-device [buffer]
    (-> (registry/driver :tech.compute.cpu.driver)
        (compute/default-device)))
  drv/PDriverProvider
  (get-driver [buffer]
    (registry/driver :tech.compute.cpu.driver)))


(defmacro generic-extend-java-type
  [java-type]
  `(clojure.core/extend
       ~java-type
     drv/PBuffer
     {:sub-buffer (fn [item# offset# len#]
                    (drv/sub-buffer (unsigned/->typed-buffer item#)
                                    offset# len#))
      :alias? (fn [lhs# rhs#]
                (drv/alias? (unsigned/->typed-buffer lhs#)
                            (unsigned/->typed-buffer rhs#)))
      :partially-alias? (fn [lhs# rhs#]
                          (drv/partially-alias? (unsigned/->typed-buffer lhs#)
                                                (unsigned/->typed-buffer rhs#)))}
     drv/PDeviceProvider
     {:get-device (fn [item#]
                    (drv/get-device (unsigned/->typed-buffer item#)))}

     drv/PDriverProvider
     {:get-driver (fn [item#]
                    (drv/get-driver (unsigned/->typed-buffer item#)))}
     ))


(generic-extend-java-type ByteBuffer)
(generic-extend-java-type ShortBuffer)
(generic-extend-java-type IntBuffer)
(generic-extend-java-type LongBuffer)
(generic-extend-java-type FloatBuffer)
(generic-extend-java-type DoubleBuffer)



(generic-extend-java-type (Class/forName "[B"))
(generic-extend-java-type (Class/forName "[S"))
(generic-extend-java-type (Class/forName "[I"))
(generic-extend-java-type (Class/forName "[J"))
(generic-extend-java-type (Class/forName "[F"))
(generic-extend-java-type (Class/forName "[D"))
