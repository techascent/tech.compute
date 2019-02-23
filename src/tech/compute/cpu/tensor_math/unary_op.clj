(ns tech.compute.cpu.tensor-math.unary-op
  (:require [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     all-datatypes
                     ] :as nio-access]
            [tech.compute.cpu.tensor-math.addressing
             :refer [get-elem-dims->address
                     max-shape-from-dimensions]]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math.writers :as writers]
            [tech.compute.cpu.math-operands
             :refer [as-unary-op as-typed-unary-op]])
  (:import [tech.compute.cpu UnaryOp TypedUnaryOp]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defn datatype->operating-datatype
  "The unsigned datatypes operate at the next higher width in typed space"
  [datatype]
  (case datatype
    :uint8 :int16
    :uint16 :int32
    :uint32 :int64
    :uint64 :int64
    datatype))


(defmacro call-typed-custom
  [datatype item opcode]
  (let [jvm-type (datatype->operating-datatype datatype)
        opflags (if (unsigned/unsigned-datatype? datatype)
                  TypedUnaryOp/UNSIGNED
                  0)]
    (case jvm-type
      :int8 `(.byteOp ~item ~opcode ~opflags)
      :int16 `(.shortOp ~item ~opcode ~opflags)
      :int32 `(.intOp ~item ~opcode ~opflags)
      :int64 `(.longOp ~item ~opcode ~opflags)
      :float32 `(.floatOp ~item ~opcode ~opflags)
      :float64 `(.doubleOp ~item ~opcode ~opflags))))


(defmacro make-unary-reader
  [datatype x x-idx->addr x-alpha unary-op]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)]
    `(let [x-alpha# (unsigned/datatype->cast-fn :ignored ~datatype ~x-alpha)
           x# ~x
           x-idx->addr# ~x-idx->addr
           custom# (as-unary-op ~unary-op)
           typed-custom# (as-typed-unary-op ~unary-op)]
       (if (= x-alpha# (unsigned/datatype->cast-fn :ignored ~datatype 1))
         (if custom#
           (primitive/make-converter
            ~jvm-datatype
            (store-datatype-cast-fn
             ~datatype
             (.op custom#
                  (read-datatype-cast-fn
                   ~datatype
                   (b-get x# (.idx_to_address x-idx->addr# ~'idx))))))
           (primitive/make-converter
            ~jvm-datatype
            (store-datatype-cast-fn
             ~datatype
             (call-typed-custom
              ~datatype
              typed-custom#
              (read-datatype-cast-fn
               ~datatype
               (b-get x# (.idx_to_address x-idx->addr# ~'idx)))))))
         (if custom#
           (primitive/make-converter
            ~jvm-datatype
            (store-datatype-cast-fn
             ~datatype
             (.op custom#
                  (* x-alpha#
                   (read-datatype-cast-fn
                    ~datatype
                    (b-get x# (.idx_to_address x-idx->addr# ~'idx)))))))
           (primitive/make-converter
            ~jvm-datatype
            (store-datatype-cast-fn
             ~datatype
             (call-typed-custom
              ~datatype
              typed-custom#
              (* (read-datatype-cast-fn
                  ~datatype
                  (b-get x# (.idx_to_address x-idx->addr# ~'idx)))
                 x-alpha#)))))))))


(defmacro ^:private custom-accum!-impl
  [datatype]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims# dest-alpha#
          n-elems# unary-op#]
       (let [n-elems# (long n-elems#)
             dest# (item->typed-nio-buffer ~datatype dest#)
             dest-idx->address# (get-elem-dims->address dest-dims#
                                                        (get dest-dims# :shape))
             max-shape# (max-shape-from-dimensions dest-dims#)
             writer# (writers/get-serial-writer ~jvm-datatype)]
         (writer# dest# dest-dims# max-shape# n-elems#
                  (make-unary-reader ~datatype dest# dest-idx->address#
                                     dest-alpha# unary-op#))))))


(defmacro ^:private custom-unary-op!-impl
  [datatype]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [dest# dest-dims#
          x# x-dims# x-alpha#
          n-elems# unary-op#]
       (let [n-elems# (long n-elems#)
             max-shape# (max-shape-from-dimensions dest-dims# x-dims#)
             x# (item->typed-nio-buffer ~datatype x#)
             x-idx->address# (get-elem-dims->address x-dims# max-shape#)
             writer# (writers/get-parallel-writer ~jvm-datatype)]
         (writer# dest# dest-dims# max-shape# n-elems#
                  (make-unary-reader ~datatype x# x-idx->address#
                                     x-alpha# unary-op#))))))


(defmacro make-custom-unary-ops
  []
  (->> (for [dtype all-datatypes]
         [[dtype :custom] {:unary-accum! `(custom-accum!-impl ~dtype)
                           :unary-op! `(custom-unary-op!-impl ~dtype)}])
       (into {})))


(def custom-unary-ops (make-custom-unary-ops))


(def unary-op-table custom-unary-ops)
