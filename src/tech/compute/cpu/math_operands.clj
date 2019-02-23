(ns tech.compute.cpu.math-operands
  (:require [tech.datatype :as dtype])
  (:import [tech.compute.cpu
            UnaryOp TypedUnaryOp
            BinaryOp TypedBinaryOp
            UnaryReduce]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)



;; unary, binary, and reduction operators are stored in the operand table.
(def ^:dynamic *operand-table* (atom {}))


(defn add-unary-op!
  [keywd unary-op]
  (when-not (or (instance? UnaryOp unary-op)
                (instance? TypedUnaryOp unary-op))
    (throw (ex-info "Operand is not a tech.compute.cpu.UnaryOp"
                    {:operand unary-op})))
  (swap! *operand-table* assoc [keywd :unary] unary-op)
  keywd)


(defn add-binary-op!
  [keywd binary-op]
  (when-not (or (instance? BinaryOp binary-op)
                (instance? TypedBinaryOp binary-op))
    (throw (ex-info "Operand is not a tech.compute.cpu.BinaryOp"
                    {:operand binary-op})))
  (swap! *operand-table* assoc [keywd :binary] binary-op)
  keywd)


(defn add-unary-reduce!
  [keywd unary-reduce-op]
  (when-not (or (instance? UnaryReduce unary-reduce-op)
                (instance? BinaryOp unary-reduce-op)
                (instance? TypedBinaryOp unary-reduce-op))
    (throw (ex-info "Operand is not a tech.compute.cpu.UnaryReduceOp
nor a BinaryOp nor a TypedBinaryOp."
                    {:operand unary-reduce-op})))
  (let [reduce-op
        (cond
          (instance? UnaryReduce unary-reduce-op)
          unary-reduce-op
          (instance? BinaryOp unary-reduce-op)
          (let [^BinaryOp bin-op unary-reduce-op]
            (reify UnaryReduce
              (initialize [this nv] nv)
              (update [this accum nv]
                (.op bin-op accum nv))
              (finalize [this accum n-elems]
                accum)))
          (instance? TypedBinaryOp unary-reduce-op)
          (let [^TypedBinaryOp bin-op unary-reduce-op]
            (reify UnaryReduce
              (initialize [this nv] nv)
              (update [this accum nv]
                (.doubleOp bin-op accum nv 0))
              (finalize [this accum n-elems]
                accum)))
          :else
          (throw (ex-info "Failed to discern op type" {})))]
    (swap! *operand-table* assoc [keywd :unary-reduce] reduce-op))
  keywd)


(defn get-operand-of-type
  [op-kwd op-type]
  (if-let [retval (get @*operand-table* [op-kwd op-type])]
    retval
    (throw (ex-info (format "Failed to find %s operand %s"
                            (name op-type) (name op-kwd))))))


(defn get-unary-operand
  [op-kwd]
  (get-operand-of-type op-kwd :unary))

(defn get-binary-operand
  [op-kwd]
  (get-operand-of-type op-kwd :binary))

(defn get-unary-reduce-operand
  [op-kwd]
  (get-operand-of-type op-kwd :unary-reduce))


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
   :not (create-unary-op (if (= 0.0 x) 1.0 0.0))
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


(doseq [[opname impl] builtin-unary-ops]
  (add-unary-op! opname impl))



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
   :and (implement-binary-op (if (and (not= 0.0 x)
                                      (not= 0.0 y))
                               1.0 0.0))
   :or (implement-binary-op (if (or (not= 0.0 x)
                                    (not= 0.0 y))
                              1.0 0.0))
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
   :not-eq (implement-typed-binary-op (if (not= x y) 1 0))
   :> (implement-typed-binary-op (if (> x y) 1 0))
   :>= (implement-typed-binary-op (if (>= x y) 1 0))
   :< (implement-typed-binary-op (if (< x y) 1 0))
   :<= (implement-typed-binary-op (if (<= x y) 1 0))
   :atan2 (implement-binary-op (Math/atan2 x y))
   :hypot (implement-binary-op (Math/hypot x y))
   :ieee-remainder (implement-binary-op (Math/IEEEremainder x y))
   })


(defn as-binary-op
  ^BinaryOp [item]
  (when (instance? BinaryOp item)
    item))


(defn as-typed-binary-op
  ^TypedBinaryOp [item]
  (when (instance? TypedBinaryOp item)
    item))


(doseq [[opname impl] builtin-binary-ops]
  (add-binary-op! opname impl))


(def builtin-reduce-ops
  {:mean (reify UnaryReduce
           (initialize [this fv]
             fv)
           (update [this accum nv]
             (+ accum nv))
           (finalize [this accum ne]
             (/ accum (double ne))))
   :magnitude (reify UnaryReduce
                (initialize [this fv]
                  (* fv fv))
                (update [this accum nv]
                  (+ accum (* nv nv)))
                (finalize [this accum ne]
                  (Math/sqrt (double accum))))
   :magnitude-squared (reify UnaryReduce
                        (initialize [this fv]
                          (* fv fv))
                        (update [this accum nv]
                          (+ accum (* nv nv)))
                        (finalize [this accum ne]
                          accum))})


(doseq [[opname red-impl] builtin-reduce-ops]
  (add-unary-reduce! opname red-impl))


(doseq [[opname bin-impl] builtin-binary-ops]
  (add-unary-reduce! opname bin-impl)
  (add-unary-reduce! :sum (get builtin-binary-ops :+)))


(defn builtin-operator-list
  [operand-type]
  (->> @*operand-table*
       (filter #(= operand-type (second (first %))))
       (map ffirst)
       (remove #(= :custom %))
       distinct
       sort))
