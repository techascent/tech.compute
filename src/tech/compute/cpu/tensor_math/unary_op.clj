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
            [tech.compute.cpu.tensor-math.writers :as writers])
  (:import [tech.compute.cpu UnaryOp TypedUnaryOp]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defmacro create-unary-op
  [op-code]
  `(reify UnaryOp
     (op [this# ~'x]
       (double ~op-code))))


(defmacro float-double-unary-op
  [double-op-code float-op-code]
  `(reify TypedUnaryOp
     (doubleOp [this# ~'x flags#]
       ~double-op-code)
     (floatOp [this# ~'x flags#]
       ~float-op-code)
     (longOp [this# x# flags#]
       (-> (.doubleOp this# (unchecked-double x#) 0)
           unchecked-long))
     (intOp [this# x# flags#]
       (-> (.doubleOp this# (unchecked-double x#) 0)
           unchecked-int))
     (shortOp [this# x# flags#]
       (-> (.doubleOp this# (unchecked-float x#) 0)
           unchecked-short))
     (byteOp [this# x# flags#]
       (-> (.doubleOp this# (unchecked-float x#) 0)
           unchecked-byte))))

(defmacro long-unary-op
  [long-op-code]
  `(reify TypedUnaryOp
     (doubleOp [this# x# flags#]
       (-> (.longOp this# (unchecked-long x#) flags#)
           unchecked-double))
     (floatOp [this# x# flags#]
       (-> (.longOp this# (unchecked-long x#) flags#)
           unchecked-float))
     (longOp [this# ~'x ~'flags]
       ~long-op-code)
     (intOp [this# x# flags#]
       (-> (.longOp this# (unchecked-long x#) 0)
           unchecked-int))
     (shortOp [this# x# flags#]
       (-> (.shortOp this# (unchecked-long x#) 0)
           unchecked-short))
     (byteOp [this# x# flags#]
       (-> (.doubleOp this# (unchecked-long x#) 0)
           unchecked-byte))))

(defmacro typed-unary-op
  [op-code]
  `(reify TypedUnaryOp
     (doubleOp [this# ~'x ~'flags]
       (unchecked-double ~op-code))
     (floatOp [this# ~'x ~'flags]
       (unchecked-float ~op-code))
     (longOp [this# ~'x ~'flags]
       (unchecked-long ~op-code))
     (intOp [this# ~'x ~'flags]
       (unchecked-int ~op-code))
     (shortOp [this# ~'x ~'flags]
       (unchecked-short ~op-code))
     (byteOp [this# ~'x ~'flags]
       (unchecked-byte~op-code))))


(def builtin-unary-ops
  {:floor (create-unary-op (Math/floor x))
   :ceil (create-unary-op (Math/ceil x))
   :round (create-unary-op (Math/round x))
   :rint (create-unary-op (Math/rint x))
   :- (create-unary-op (- x))
   :logistic (create-unary-op
              (/ 1.0
                 (+ 1.0 (Math/exp (- x)))))
   :exp (create-unary-op (Math/exp x))
   :expm1 (create-unary-op (Math/expm1 x))
   :log (create-unary-op (Math/log x))
   :log10 (create-unary-op (Math/log10 x))
   :log1p (create-unary-op (Math/log1p x))
   :signum (create-unary-op (Math/signum x))
   :sqrt (create-unary-op (Math/sqrt x))
   :cbrt (create-unary-op (Math/cbrt x))
   :abs (create-unary-op (Math/abs x))
   :sq (typed-unary-op (unchecked-multiply x x))
   :sin (create-unary-op (Math/sin x))
   :sinh (create-unary-op (Math/sinh x))
   :cos (create-unary-op (Math/cos x))
   :cosh (create-unary-op (Math/cosh x))
   :tan (create-unary-op (Math/tan x))
   :tanh (create-unary-op (Math/tanh x))
   :acos (create-unary-op (Math/acos x))
   :asin (create-unary-op (Math/asin x))
   :atan (create-unary-op (Math/atan x))
   :to-degrees (create-unary-op (Math/toDegrees x))
   :to-radians (create-unary-op (Math/toRadians x))
   :next-up (float-double-unary-op (Math/nextUp x) (Math/nextUp x))
   :next-down (float-double-unary-op (Math/nextDown x) (Math/nextDown x))
   :ulp (float-double-unary-op (Math/ulp x) (Math/ulp x))
   ;;This is strained.  It would need a different interface really.
   :bit-not (long-unary-op (bit-not x))
   :/ (typed-unary-op (/ x))
   :noop (create-unary-op x)})


(defn as-unary-op
  ^UnaryOp [item]
  (when (instance? UnaryOp item)
    item))


(defn as-typed-unary-op
  ^TypedUnaryOp [item]
  (when (instance? TypedUnaryOp item)
    item))


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
             dest-alpha# (unsigned/datatype->cast-fn :ignored ~datatype dest-alpha#)
             custom# (as-unary-op unary-op#)
             typed-custom# (as-typed-unary-op unary-op#)
             writer# (writers/get-serial-writer ~jvm-datatype)]
         (if custom#
           (writer# dest# dest-dims# max-shape# n-elems#
                    (primitive/make-converter
                     ~jvm-datatype
                     (store-datatype-cast-fn
                      ~datatype
                      (.op custom#
                           (*
                            (read-datatype-cast-fn
                             ~datatype
                             (b-get dest# (.idx_to_address dest-idx->address#
                                                           ~'idx)))
                            dest-alpha#)))))
           (writer# dest# dest-dims# max-shape# n-elems#
                    (primitive/make-converter
                     ~jvm-datatype
                     (store-datatype-cast-fn
                      ~datatype
                      (call-typed-custom
                       ~datatype typed-custom#
                       (*
                        (read-datatype-cast-fn
                         ~datatype
                         (b-get dest# (.idx_to_address dest-idx->address#
                                                       ~'idx)))
                        dest-alpha#))))))))))


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
             x-alpha# (unsigned/datatype->cast-fn :ignored ~datatype x-alpha#)
             ^UnaryOp custom-op# unary-op#
             writer# (writers/get-parallel-writer ~jvm-datatype)]
         (writer# dest# dest-dims# max-shape# n-elems#
                  (primitive/make-converter
                   ~jvm-datatype
                   (store-datatype-cast-fn
                    ~datatype
                    (.op custom-op#
                         (* (read-datatype-cast-fn
                             ~datatype
                             (b-get x# (.idx_to_address x-idx->address# ~'idx)))
                            x-alpha#)))))))))


(defmacro make-custom-unary-ops
  []
  (->> (for [dtype all-datatypes]
         [[dtype :custom] {:unary-accum! `(custom-accum!-impl ~dtype)
                           :unary-op! `(custom-unary-op!-impl ~dtype)}])
       (into {})))


(def custom-unary-ops (make-custom-unary-ops))


(def unary-op-table
  (merge custom-unary-ops
         (->> (for [dtype all-datatypes
                    [op-name operator] builtin-unary-ops]
                (let [{:keys [unary-accum! unary-op!]}
                      (get custom-unary-ops [dtype :custom])]
                  [[dtype op-name]
                   {:unary-op!
                    (fn [dest dest-dims
                         x x-dims x-alpha
                         n-elems]
                      (unary-op! dest dest-dims x x-dims x-alpha
                                 n-elems operator))
                    :unary-accum!
                    (fn [dest dest-dims dest-alpha
                         n-elems]
                      (unary-accum! dest dest-dims dest-alpha
                                    n-elems operator))}]))
              (into {}))))
