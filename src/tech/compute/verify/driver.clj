(ns tech.compute.verify.driver
  (:require [clojure.test :refer :all]
            [tech.compute.driver :as drv]
            [tech.datatype :as dtype]
            [tech.datatype.base :as dtype-base]
            [clojure.core.matrix :as m]
            [tech.compute.verify.utils :as verify-utils]
            [tech.compute :as compute]
            [tech.datatype.java-unsigned :as unsigned]))



(defn simple-stream
  [driver datatype]
  (verify-utils/with-default-device-and-stream
    driver
    (let [buf-a (compute/allocate-host-buffer driver 10 datatype)
          output-buf-a (compute/allocate-host-buffer driver 10 datatype)
          buf-b (compute/allocate-device-buffer device 10 datatype)
          jvm-dtype  (unsigned/datatype->jvm-datatype datatype)
          input-data (dtype/make-array-of-type jvm-dtype (range 10))
          output-data (dtype/make-array-of-type jvm-dtype 10)]
      (dtype/copy! input-data 0 buf-a 0 10)
      (dtype-base/set-value! buf-a 0 100.0)
      (dtype/copy! buf-a 0 output-data 0 10)
      (compute/copy-host->device buf-a 0 buf-b 0 10)
      (compute/copy-device->host buf-b 0 output-buf-a 0 10)
      (compute/sync-with-host stream {:gc-roots [buf-a buf-b]})
      (dtype/copy! output-buf-a 0 output-data 0 10)
      (is (= [100.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0]
             (mapv double output-data))))))
