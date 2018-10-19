(ns tech.compute.cpu.tensor-test
  (:require [tech.compute.verify.tensor :as verify-tensor]
            [tech.compute.verify.utils
             :refer [def-double-float-test
                     def-all-dtype-test
                     *datatype*
                     def-int-long-test
                     test-wrapper
                     def-all-dtype-exception-unsigned]]
            [clojure.test :refer :all]
            [tech.compute.cpu.driver :refer [driver]]
            [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.compute.tensor :as ct]
            [tech.datatype.core :as dtype]))


(use-fixtures :each test-wrapper)


(def-all-dtype-test assign-constant!
  (verify-tensor/assign-constant! (driver) *datatype*))


(def-all-dtype-test assign-marshal
  (verify-tensor/assign-marshal (driver) *datatype*))


(def-all-dtype-exception-unsigned binary-constant-op
  (verify-tensor/binary-constant-op (driver) *datatype*))


(def-all-dtype-exception-unsigned binary-op
  (verify-tensor/binary-op (driver) *datatype* ))


(def-all-dtype-test unary-op
  (verify-tensor/unary-op (driver) *datatype*))


(def-all-dtype-test channel-op
  (verify-tensor/channel-op (driver) *datatype*))


(def-double-float-test gemm
  (verify-tensor/gemm (driver) *datatype*))


(def-double-float-test gemv
  (verify-tensor/gemv (driver) *datatype*))


(def-all-dtype-exception-unsigned ternary-op-select
  (verify-tensor/ternary-op-select (driver) *datatype*))


(def-all-dtype-test unary-reduce
  (verify-tensor/unary-reduce (driver) *datatype*))


(def-all-dtype-test transpose
  (verify-tensor/transpose (driver) *datatype*))


(def-int-long-test mask
  (verify-tensor/mask (driver) *datatype*))


(def-all-dtype-test select
  (verify-tensor/select (driver) *datatype*))


(def-all-dtype-test select-with-persistent-vectors
  (verify-tensor/select-with-persistent-vectors (driver) *datatype*))


(def-all-dtype-test select-transpose-interaction
  (verify-tensor/select-transpose-interaction (driver) *datatype*))


;;Note that this is not a float-double test.
(deftest rand-operator
  (verify-tensor/rand-operator (driver) :float32))


(def-all-dtype-test indexed-tensor
  (verify-tensor/indexed-tensor (driver) *datatype*))


(def-double-float-test magnitude-and-mag-squared
  (verify-tensor/magnitude-and-mag-squared (driver) *datatype*))


(def-double-float-test constrain-inside-hypersphere
  (verify-tensor/constrain-inside-hypersphere (driver) *datatype*))


(deftest default-tensor-stream
  (testing "Default stream binding; main thread cpu stream."
    (ct/enable-cpu-tensors!)
    (let [tens-a (ct/->tensor [1 2 3])
          double-data (ct/to-double-array tens-a)]
      (is (= [1.0 2.0 3.0]
             (vec double-data))))))


(deftest infer-stream
  (is (= [1 2 3]
         (-> (cpu-tm/typed-bufferable->tensor
              (dtype/make-array-of-type :int16 [1 2 3]))
             ct/clone
             (ct/to-jvm :datatype :int32)))))
