(ns tech.compute.verify.utils
  (:require [clojure.core.matrix :as m]
            [tech.resource :as resource]
            [clojure.test :refer :all]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.datatype.java-primitive :as primitive]
            [tech.compute :as compute])
  (:import [java.math BigDecimal MathContext]))


(defn test-wrapper
  [test-fn]
  (resource/with-resource-context
    ;;Turn on if you want much slower tests.
    (with-bindings {#'resource/*resource-debug-double-free* false}
      (test-fn))))


(defmacro with-default-device-and-stream
  [driver & body]
  `(resource/with-resource-context
     (let [~'device (compute/default-device ~driver)
           ~'stream (compute/default-stream ~'device)]
       ~@body)))


(def ^:dynamic *datatype* :float64)


(defmacro datatype-list-tests
  [datatype-list test-name & body]
  `(do
     ~@(for [datatype datatype-list]
         (do
           `(deftest ~(symbol (str test-name "-" (name datatype)))
              (with-bindings {#'*datatype* ~datatype}
                ~@body))))))



(defmacro def-double-float-test
  [test-name & body]
  `(datatype-list-tests [:float64 :float32] ~test-name ~@body))


(defmacro def-int-long-test
  [test-name & body]
  `(datatype-list-tests [:int32 :uint32 :int64 :uint64]
                        ~test-name
                        ~@body))


(defmacro def-all-dtype-test
  [test-name & body]
  `(datatype-list-tests ~unsigned/datatypes ~test-name ~@body))


(defmacro def-all-dtype-exception-unsigned
  "Some platforms can detect unsigned errors."
  [test-name & body]
  `(do
     (datatype-list-tests ~primitive/datatypes ~test-name ~@body)
     (datatype-list-tests ~unsigned/unsigned-datatypes ~test-name
                          (is (try
                                ~@body
                                nil
                                (catch Throwable e# e#))))))
