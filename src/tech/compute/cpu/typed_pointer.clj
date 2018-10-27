(ns tech.compute.cpu.typed-pointer
  (:require [tech.datatype.jna :as dtype-jna]
            [tech.compute.driver :as drv]
            [tech.compute :as compute]
            [tech.datatype :as dtype]
            [tech.compute.cpu.utils :as cpu-utils]
            [tech.resource :as resource]
            [tech.compute.registry :as registry]
            [tech.datatype.javacpp :as jcpp-dtype])
  (:import [tech.datatype.jna TypedPointer]
           [com.sun.jna Pointer]
           [org.bytedeco.javacpp.Pointer]))



(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defn- thing->addr
  ^long [item]
  (-> (dtype-jna/->ptr-backing-store item)
      dtype-jna/pointer->address))


(defn maybe->typed-pointer
  [item]
  (if (satisfies? dtype-jna/PToPtr item)
    (or (dtype-jna/as-typed-pointer item)
        (dtype-jna/->typed-pointer item))))


(extend-type TypedPointer
  drv/PBuffer
  (sub-buffer [buffer offset length]
    (let [datatype (dtype/get-datatype buffer)
          addr (thing->addr buffer)
          elem-count (dtype/ecount buffer)
          byte-size (dtype/datatype->byte-size datatype)
          offset (long offset)
          length (long length)
          _ (when-not (<= elem-count
                          (+ offset length))
              (throw (ex-info "Offset out of range"
                              {:elem-count elem-count
                               :offset offset
                               :length length})))
          byte-offset (* offset byte-size)
          new-addr (+ addr byte-offset)
          new-ptr (dtype-jna/make-jna-pointer new-addr)]
      (dtype-jna/->TypedPointer new-ptr (* length byte-size) datatype)))

  (alias? [lhs rhs]
    (when-let [rhs (maybe->typed-pointer rhs)]
      (let [lhs-addr (thing->addr lhs)
            rhs-addr (thing->addr rhs)]
        (= lhs-addr rhs-addr))))

  (partially-alias? [lhs rhs]
    (when-let [rhs (maybe->typed-pointer rhs)]
      (let [lhs-addr (thing->addr lhs)
            rhs-addr (thing->addr rhs)]
        (cpu-utils/in-range? lhs-addr (dtype/ecount lhs)
                             rhs-addr (dtype/ecount rhs)))))
  ;;For uniformity all host/device buffers must implement the resource protocol.
  resource/PResource
  (release-resource [_])
  drv/PDeviceProvider
  (get-device [buffer]
    (-> (drv/get-driver buffer)
        (compute/default-device)))
  drv/PDriverProvider
  (get-driver [buffer]
    (-> (registry/cpu-driver-name)
        registry/driver)))


(defmacro extend-typed-pointer-like-type
  [newtype]
  `(clojure.core/extend
       ~newtype
     drv/PBuffer
     {:sub-buffer (fn [item# offset# len#]
                    (drv/sub-buffer (dtype-jna/->typed-pointer item#)
                                    offset# len#))
      :alias? (fn [lhs# rhs#]
                (drv/alias? (dtype-jna/->typed-pointer lhs#)
                            rhs#))
      :partially-alias? (fn [lhs# rhs#]
                         (drv/partially-alias? (dtype-jna/->typed-pointer lhs#)
                                               rhs#))}
     drv/PDeviceProvider
     {:get-device (fn [item#]
                    (drv/get-device (dtype-jna/->typed-pointer item#)))}

     drv/PDriverProvider
     {:get-driver (fn [item#]
                    (drv/get-driver (dtype-jna/->typed-pointer item#)))}))


(extend-typed-pointer-like-type org.bytedeco.javacpp.Pointer)
(extend-typed-pointer-like-type Pointer)
