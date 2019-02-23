(ns tech.compute.tensor.functional-test
  (:require [tech.compute.tensor.functional :as functional]
            [tech.compute.cpu.math-operands :as math-ops]
            [tech.compute.tensor.functional.impl :as func-impl]
            [tech.datatype :as dtype]
            [clojure.test :refer :all]))


(deftest scalar-booleans-return-boolean-values
  (is (= true (functional/> 5 2))))


(deftest boolean-values-make-sense
  (let [src-ary (range 5)]
    (is (= (mapv #(> 1 %) src-ary)
           (-> (functional/> 1 (double-array src-ary))
               (dtype/copy! (boolean-array 5))
               vec)))))


(deftest register-namespaced-symbols
  (testing "Test that you can register a namespaced function"
    (let [local-lang-atom (atom {})]
      (math-ops/add-unary-op! :tech.compute.testing/test-fn
                              (math-ops/create-unary-op
                               (* 2 x)))
      (func-impl/make-math-fn local-lang-atom
                              :tech.compute.testing/test-fn
                              #{:unary})
      (is (functional/eq 8
                         (func-impl/eval-expr
                          {:symbol-map @local-lang-atom} '(test-fn 4)))))))


(deftest binary-symbols-extremely-flexible
  (is (= 14
         (functional/max (int-array (range 5 15))))))
