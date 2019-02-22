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


(defmacro implement-binary-op
  [opcode]
  `(reify BinaryOp
     (op [this# ~'x ~'y]
       ~opcode)))


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

(defmacro long-binary-op
  [opcode]
  `(reify TypedBinaryOp
     (doubleOp [this# x# y# flags#]
       (-> (.longOp this#
                    (unchecked-long x#)
                    (unchecked-long y#)
                    flags#)
           unchecked-double))
     (floatOp [this# x# y# flags#]
       (-> (.longOp this#
                    (unchecked-long x#)
                    (unchecked-long y#)
                    flags#)
           unchecked-float))
     (longOp [this# ~'x ~'y ~'flags]
       (unchecked-long ~opcode))
     (intOp [this# x# y# flags#]
       (-> (.longOp this#
                    (unchecked-long x#)
                    (unchecked-long y#)
                    flags#)
           unchecked-int))
     (shortOp [this# x# y# flags#]
       (-> (.longOp this#
                    (unchecked-long x#)
                    (unchecked-long y#)
                    flags#)
           unchecked-short))
     (byteOp [this# x# y# flags#]
       (-> (.longOp this#
                    (unchecked-long x#)
                    (unchecked-long y#)
                    flags#)
           unchecked-byte))))


(def builtin-binary-ops
  {
   :+ (implement-typed-binary-op (+ x y))
   :- (implement-typed-binary-op (- x y))
   :/ (implement-typed-binary-op (/ x y))
   :* (implement-typed-binary-op (* x y))
   :rem (long-binary-op (Math/floorMod x y))
   :quot (long-binary-op (Math/floorDiv x y))
   :pow (implement-binary-op (Math/pow x y))
   ;;Math/max and friends aren't defined for all primitives leading to reflection
   ;;warnings.
   :max (implement-typed-binary-op (if (> x y) x y))
   :min (implement-typed-binary-op (if (> x y) y x))
   :bit-and (long-binary-op (bit-and x y))
   :bit-and-not (long-binary-op (bit-and-not x y))
   :bit-or (long-binary-op (bit-or x y))
   :bit-xor (long-binary-op (bit-xor x y))
   :bit-clear (long-binary-op (bit-clear x y))
   :bit-flip (long-binary-op (bit-flip x y))
   :bit-test (long-binary-op (bit-test x y))
   :bit-set (long-binary-op (bit-set x y))
   :bit-shift-left (long-binary-op (bit-shift-left x y))
   :bit-shift-right (long-binary-op (bit-shift-right x y))
   :unsigned-bit-shift-right (long-binary-op (unsigned-bit-shift-right x y))
   :eq (implement-typed-binary-op (if (= x y) 1 0))
   :> (implement-typed-binary-op (if (> x y) 1 0))
   :>= (implement-typed-binary-op (if (>= x y) 1 0))
   :< (implement-typed-binary-op (if (< x y) 1 0))
   :<= (implement-typed-binary-op (if (<= x y) 1 0))
   :atan2 (implement-binary-op (Math/atan2 x y))
   :hypot (implement-binary-op (Math/hypot x y))
   :ieee-remainder (implement-binary-op (Math/IEEEremainder x y))
   })


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
