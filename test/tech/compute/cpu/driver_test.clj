(ns tech.compute.cpu.driver-test
  (:require [tech.compute.cpu.driver :as cpu]
            [tech.compute.driver :as drv]
            [tech.datatype.core :as dtype]
            [think.resource.core :as resource]
            [clojure.test :refer :all]
            [tech.compute.verify.utils :refer [def-all-dtype-test
                                                def-double-float-test] :as test-utils]
            [tech.compute.verify.driver :as verify-driver]))


(use-fixtures :each test-utils/test-wrapper)


(defn driver
  []
  (cpu/driver))


(def-all-dtype-test simple-stream
  (verify-driver/simple-stream (driver) test-utils/*datatype*))
