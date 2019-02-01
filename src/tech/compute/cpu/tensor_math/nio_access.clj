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
