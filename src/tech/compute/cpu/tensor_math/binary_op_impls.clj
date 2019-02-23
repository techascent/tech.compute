(ns tech.compute.cpu.tensor-math.binary-op-impls
  (:require [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     all-datatypes
                     datatype->cast-fn] :as nio-access]
           [tech.compute.cpu.tensor-math.writers :as writers]
           [tech.datatype.java-unsigned :as unsigned]
           [tech.datatype.java-primitive :as primitive]
           [tech.compute.cpu.math-operands
            :refer [as-binary-op
                    as-typed-binary-op]])
  (:import [tech.compute.cpu TypedBinaryOp BinaryOp]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defmacro call-typed-op
  [datatype custom-op & args]
  (case datatype
    :int8 `(.byteOp ~custom-op ~@args)
    :int16 `(.shortOp ~custom-op ~@args)
    :int32 `(.intOp ~custom-op ~@args)
    :int64 `(.longOp ~custom-op ~@args)
    :float32 `(.floatOp ~custom-op ~@args)
    :float64 `(.doubleOp ~custom-op ~@args)))


(defn datatype->operating-datatype
  "The unsigned datatypes operate at the next higher width in typed space"
  [datatype]
  (case datatype
    :uint8 :int16
    :uint16 :int32
    :uint32 :int64
    :uint64 :int64
    datatype))


(defmacro make-custom-binary-op-converter
  [datatype
   reverse-operation? custom-op
   x-expr y-expr]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)
        op-datatype (datatype->operating-datatype datatype)
        unsigned-flag (int (if (unsigned/unsigned-datatype? datatype)
                             1 0))]
    `(let [custom-op# (as-binary-op ~custom-op)
           typed-op# (as-typed-binary-op ~custom-op)]
       (if ~reverse-operation?
         (if custom-op#
           (primitive/make-converter
            ~jvm-datatype
            (store-datatype-cast-fn
             ~datatype
             (.op custom-op# ~y-expr ~x-expr)))
           (primitive/make-converter
            ~jvm-datatype
            (store-datatype-cast-fn
             ~datatype
             (call-typed-op
              ~op-datatype
              typed-op#
              ~y-expr ~x-expr ~unsigned-flag))))
         (if custom-op#
           (primitive/make-converter
            ~jvm-datatype
            (store-datatype-cast-fn
             ~datatype
             (.op custom-op# ~x-expr ~y-expr)))
           (primitive/make-converter
            ~jvm-datatype
            (store-datatype-cast-fn
             ~datatype
             (call-typed-op
              ~op-datatype
              typed-op#
              ~x-expr ~y-expr ~unsigned-flag))))))))


(defmacro make-converter-elide-modulators
  [datatype
   reverse-operation? custom-op
   x-alpha y-alpha
   x-expr y-expr]
  `(if (and (= ~x-alpha (datatype->cast-fn ~datatype 1))
            (= ~y-alpha (datatype->cast-fn ~datatype 1)))
     (make-custom-binary-op-converter
      ~datatype ~reverse-operation? ~custom-op
      ~x-expr ~y-expr)
     (make-custom-binary-op-converter
      ~datatype ~reverse-operation? ~custom-op
      (* ~x-alpha ~x-expr)
      (* ~y-alpha ~y-expr))))
