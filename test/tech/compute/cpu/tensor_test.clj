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
            [tech.compute.tensor.defaults :as ct-defaults]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.math :as ct-tm]
            [tech.compute.driver :as compute-drv]
            [tech.datatype :as dtype]
            [tech.datatype.jna :as dtype-jna]
            [clojure.core.matrix :as m]))


(use-fixtures :each test-wrapper)


(def-all-dtype-test assign-constant!
  (verify-tensor/assign-constant! (driver) :int64))


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
             (vec double-data))))
    ;;Hijacking test for test out ->jvm functionality
    (let [src-data [[1 2] [3 4]]
          tens-b (ct/->tensor src-data)]
      (is (= src-data
             (ct/to-jvm tens-b :datatype :int32))))
    (let [src-data [[1 2]]
          tens-b (ct/->tensor src-data)]
      (is (= src-data
             (ct/to-jvm tens-b :datatype :int32))))
    (let [src-data [[2]]
          tens-b (ct/->tensor src-data)]
      (is (= src-data
             (ct/to-jvm tens-b :datatype :int32))))
    (let [src-data [1 2]
          tens-b (ct/->tensor src-data)]
      (is (= src-data
             (ct/to-jvm tens-b :datatype :int32))))
    (let [src-data [[[1 2]]]
          tens-b (ct/->tensor src-data)]
      (is (= src-data
             (ct/to-jvm tens-b :datatype :int32))))))


(deftest infer-stream
  (testing "Test that the different buffer types of the cpu backend can interact reasonably"
    (is (= [1 2 3]
           (-> (dtype/make-array-of-type :int16 [1 2 3])
               ct/clone
               (ct/to-jvm :datatype :int32))))))


(deftest raw-array-ops
  (testing "Test that raw arrays can be tensors and that raw arrays are returned from basic ops"
    (let [test-tens (dtype/make-array-of-type :int16 [1 2 3])
          next-tens (dtype/make-array-of-type :int32 3)
          assign-result (ct/assign! next-tens test-tens)]
      (is (instance? (Class/forName "[I") assign-result))
      (is (instance? (Class/forName "[I") (ct/clone assign-result)))
      (is (instance? (Class/forName "[S") (ct/unary-op! test-tens 1.0 test-tens :ceil)))
      (is (instance? (Class/forName "[I") (ct/binary-op! assign-result 1.0 assign-result 1.0 next-tens :+))))))


(deftest new-ops-return-base-container-when-possible
  (testing "Test that ->tensor and new-tensor return the base container when possible."
    (let [test-tensor (ct/->tensor [1 2 3 4])]
      (is (instance? (Class/forName "[D") test-tensor)))
    (let [test-device (-> (ct-defaults/infer-stream ())
                          (compute-drv/get-device))]
      (is (compute-drv/acceptable-device-buffer? test-device (double-array 5)))
      (is (compute-drv/acceptable-device-buffer? test-device (dtype/make-buffer-of-type :float32 5)))
      (is (compute-drv/acceptable-device-buffer? test-device (dtype/make-typed-buffer :float32 5)))
      (is (compute-drv/acceptable-device-buffer? test-device (dtype-jna/make-typed-pointer :float32 5))))
    ;;We can override the base container used for the cpu system in a function call
    (let [test-tensor (ct/->tensor [1 2 3 4] :container-fn dtype-jna/make-typed-pointer)]
      (is (dtype-jna/typed-pointer? test-tensor)))
    (let [test-tensor (ct/->tensor [[1 2 3 4]] :container-fn dtype-jna/make-typed-pointer)]
      (is (ct/tensor? test-tensor))
      (is (= [1 4] (ct/shape test-tensor))))))


(def-double-float-test cholesky-decomp
  (verify-tensor/cholesky-decomp (driver) *datatype*))
