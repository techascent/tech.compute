(ns tech.compute.cpu.tensor-math.nio-access
  "Macros for fast typed access and unsigned pathways."
  (:require [tech.datatype.java-primitive :as primitive]
            [tech.datatype.java-unsigned :as unsigned]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)



(defmacro item->typed-nio-buffer
  [datatype item]
  (let [jvm-dtype (unsigned/datatype->jvm-datatype datatype)]
    `(primitive/datatype->buffer-cast-fn
      ~jvm-dtype
      (primitive/->buffer-backing-store ~item))))


(defmacro b-put
  [buffer idx value]
  `(.put ~buffer
         (+ ~idx (.position ~buffer))
         ~value))


(defmacro b-get
  [buffer idx]
  `(.get ~buffer (+ ~idx (.position ~buffer))))


(defmacro datatype->cast-fn
  [dtype val]
  `(unsigned/datatype->cast-fn :ignored ~dtype ~val))


(defmacro store-datatype-cast-fn
  [dtype val]
  (let [jvm-dtype (unsigned/datatype->jvm-datatype dtype)]
    `(unsigned/datatype->unchecked-jvm-cast-fn
      ~dtype
      ~jvm-dtype
      ~val)))


(defmacro read-datatype-cast-fn
  [dtype val]
  (let [jvm-dtype (unsigned/datatype->jvm-datatype dtype)]
    `(unsigned/datatype->unchecked-cast-fn
      ~jvm-dtype
      ~dtype
      ~val)))


(def all-datatypes (concat primitive/datatypes unsigned/unsigned-datatypes))


(defmacro datatype-iterator
  [iter-macro]
  `(vector
    ~@(for [dtype all-datatypes]
        `(vector ~dtype (~iter-macro ~dtype)))))


(defmacro unary-op-impl
  [operation x]
  (condp = operation
    :floor `(Math/floor (double ~x))
    :ceil `(Math/ceil (double ~x))
    :round `(Math/round (double ~x))
    :- `(- ~x)
    :tanh `(Math/tanh (double ~x))
    :logistic `(/ 1.0
                  (+ 1.0 (Math/exp (- ~x))))
    :exp `(Math/exp (double ~x))
    :sqrt `(Math/sqrt (double ~x))
    :noop `(double ~x)))



(defmacro binary-op-impl
  [operation x y]
  (condp = operation
    :+ `(+ ~x ~y)
    :- `(- ~x ~y)
    :/ `(/ ~x ~y)
    :* `(* ~x ~y)
    ;;Math/max and friends aren't defined for all primitives leading to reflection
    ;;warnings.
    :max `(if (> ~x ~y) ~x ~y)
    :min `(if (> ~x ~y) ~y ~x)
    :bit-and `(bit-and (unchecked-int ~x) (unchecked-int ~y))
    :bit-xor `(bit-xor (unchecked-int ~x) (unchecked-int ~y))
    :eq `(if (= ~x ~y)
           1
           0)
    :> `(if (> ~x ~y)
           1
           0)
    :>= `(if (>= ~x ~y)
           1
           0)
    :< `(if (< ~x ~y)
           1
           0)
    :<= `(if (<= ~x ~y)
           1
           0)))


(defmacro binary-op-rev-ops
  [operation reverse-operands? x y]
  (if reverse-operands?
    `(binary-op-impl ~operation ~y ~x)
    `(binary-op-impl ~operation ~x ~y)))
