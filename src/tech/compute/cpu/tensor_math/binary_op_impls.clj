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
           [tech.datatype.java-primitive :as primitive])
  (:import [tech.compute.cpu TypedBinaryOp BinaryOp]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defmacro implement-typed-binary-op
  [opcode]
  `(reify TypedBinaryOp
     (doubleOp [this# ~'x ~'y ~'flags]
       (unchecked-double ~opcode))
     (floatOp [this# ~'x ~'y ~'flags]
       (unchecked-float ~opcode))
     (longOp [this# ~'x ~'y ~'flags]
       (unchecked-long ~opcode))
     (intOp [this# ~'x ~'y ~'flags]
       (unchecked-int ~opcode))
     (shortOp [this# ~'x ~'y ~'flags]
       (unchecked-short ~opcode))
     (byteOp [this# ~'x ~'y ~'flags]
       (unchecked-byte ~opcode))))


(def builtin-binary-ops
  {
   :+ (implement-typed-binary-op (+ x y))
   :- (implement-typed-binary-op (- x y))
   :/ (implement-typed-binary-op (/ x y))
   :* (implement-typed-binary-op (* x y))
   ;;Math/max and friends aren't defined for all primitives leading to reflection
   ;;warnings.
   :max (implement-typed-binary-op (if (> x y) x y))
   :min (implement-typed-binary-op (if (> x y) y x))
   :bit-and (implement-typed-binary-op (bit-and (unchecked-int x) (unchecked-int y)))
   :bit-xor (implement-typed-binary-op (bit-xor (unchecked-int x) (unchecked-int y)))
   :eq (implement-typed-binary-op (if (= x y)
                                    1
                                    0))
   :> (implement-typed-binary-op (if (> x y)
                                   1
                                   0))
   :>= (implement-typed-binary-op (if (>= x y)
                                    1
                                    0))
   :< (implement-typed-binary-op (if (< x y)
                                   1
                                   0))
   :<= (implement-typed-binary-op (if (<= x y)
                                    1
                                    0))})

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


(defn ->base-binary-operator
  ^BinaryOp [item]
  (when (instance? BinaryOp item)
    item))


(defn ->typed-binary-operator
  ^TypedBinaryOp [item]
  (when (instance? TypedBinaryOp item)
    item))


(defmacro make-custom-binary-op-converter
  [datatype
   reverse-operation? custom-op
   x-expr y-expr]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)
        op-datatype (datatype->operating-datatype datatype)
        unsigned-flag (int (if (unsigned/unsigned-datatype? datatype)
                             1 0))]
    `(let [custom-op# (->base-binary-operator ~custom-op)
           typed-op# (->typed-binary-operator ~custom-op)]
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
              (->typed-binary-operator ~custom-op)
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
              (->typed-binary-operator ~custom-op)
              ~x-expr ~y-expr ~unsigned-flag))))))))
