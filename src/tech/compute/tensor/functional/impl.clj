(ns tech.compute.tensor.functional.impl
  "Implementation details of the tensor functional interface"
  (:require [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math.unary-op :as unary-op]
            [tech.compute.cpu.tensor-math.binary-op-impls :as binop-impls]
            [tech.compute.cpu.tensor-math.unary-reduce :as unary-reduce]
            [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.datatype :as dtype])
  (:import [tech.compute.cpu UnaryOp TypedUnaryOp
            BinaryOp TypedBinaryOp
            UnaryReduce]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defn- as-un-op
  ^UnaryOp [item]
  (when (instance? UnaryOp item)
    item))

(defn- as-typed-un-op
  ^TypedUnaryOp [item]
  (when (instance? TypedUnaryOp item)
    item))


(defn- as-bin-op
  ^BinaryOp [item]
  (when (instance? BinaryOp item)
    item))


(defn- as-typed-bin-op
  ^TypedBinaryOp [item]
  (when (instance? TypedBinaryOp item)
    item))


(defn- as-unary-red-op
  ^UnaryReduce [item]
  (when (instance? UnaryReduce item)
    item))


(defn unary-op
  [op-kwd op-arg]
  (if (number? op-arg)
    (if-let [un-op-base (or (get unary-op/builtin-unary-ops op-kwd)
                            (get @cpu-tm/custom-op-table [op-kwd :unary]))]
      (if-let [un-op (as-un-op un-op-base)]
        (.op un-op (double op-arg))
        (let [un-op (as-typed-un-op un-op-base)]
          (case (dtype/get-datatype op-arg)
            :int8 (.byteOp un-op (unchecked-byte op-arg) 0)
            :int16 (.shortOp un-op (unchecked-short op-arg) 0)
            :int32 (.intOp un-op (unchecked-int op-arg) 0)
            :int64 (.longOp un-op (unchecked-long op-arg) 0)
            :float32 (.floatOp un-op (unchecked-float op-arg) 0)
            :float64 (.doubleOp un-op (unchecked-double op-arg) 0))))
      (throw (ex-info (format "Unable to find scalar %s" op-kwd))))
    (ct/unary-op! (ct/clone op-arg) 1.0 op-arg op-kwd)))


(defn unary-reduce
  [op-kwd op-arg]
  (let [arg-shape (ct/shape op-arg)
        accum-shape (->> (concat (butlast arg-shape)
                                 [1])
                         vec)
        retval (ct/unary-reduce! (ct/from-prototype op-arg :shape accum-shape)
                                 1.0 op-arg op-kwd)]
    (if (= [1] (ct/shape retval))
      (dtype/get-value retval 0)
      retval)))


(defn- scalar-binary-op
  [op-kwd arglist]
  (if-let [untyped-bin-op (or (get binop-impls/builtin-binary-ops op-kwd)
                              (get @cpu-tm/custom-op-table [op-kwd :binary]))]
    (if-let [bin-op (as-bin-op untyped-bin-op)]
      (loop [lhs (unchecked-double (first arglist))
             arglist (rest arglist)]
        (if-let [rhs (first arglist)]
          (recur (.op bin-op (unchecked-double lhs) (unchecked-double rhs))
                 (rest arglist))
          lhs))
      (let [bin-op (as-typed-bin-op untyped-bin-op)]
        (loop [lhs (first arglist)
               arglist (rest arglist)]
          (if-let [rhs (first arglist)]
            (recur
             (case (dtype/get-datatype lhs)
               :int8 (.byteOp bin-op (unchecked-byte lhs) (unchecked-byte rhs) 0)
               :int16 (.shortOp bin-op (unchecked-short lhs) (unchecked-short rhs) 0)
               :int32 (.intOp bin-op (unchecked-int lhs) (unchecked-int rhs) 0)
               :int64 (.longOp bin-op (unchecked-long lhs) (unchecked-long rhs) 0)
               :float32 (.floatOp bin-op (unchecked-float lhs) (unchecked-float rhs) 0)
               :float64 (.doubleOp bin-op (unchecked-double lhs)
                                   (unchecked-double rhs) 0))
             (rest arglist))
            lhs))))
    (throw (ex-info (format "Unable to find implementation of op %s"
                            op-kwd)))))


(defn binary-op
  [op-kwd arglist]
  (when-not (>= (count arglist) 2)
    (throw (ex-info "")))
  (if-let [tens-seq (->> arglist
                         (filter #(or (ct/acceptable-tensor-buffer? %)
                                      (ct/tensor? %)))
                         seq)]
    (let [accum (ct/assign! (ct/from-prototype (first tens-seq))
                            (first arglist))]
      (doseq [rhs (rest arglist)]
        (ct/binary-op! accum 1.0 accum 1.0 rhs op-kwd))
      accum)
    (scalar-binary-op op-kwd arglist)))


(defonce ^:dynamic *registered-language-fns* (atom {}))


(defn register-symbol!
  [sym-name sym-value]
  (-> (swap! *registered-language-fns* assoc sym-name sym-value)
      keys))


(defn get-operand
  "Return a map of (at least)
  {:type op-type
   :operand op-fn
  }"
  [{:keys [symbol-map] :as env} op-kwd]
  (if-let [item-val (get symbol-map op-kwd)]
    item-val
    (if-let [retval (get @*registered-language-fns* op-kwd)]
      retval
      (throw (ex-info (format "Failed to find math operand: %s" op-kwd)
                      {:operand op-kwd})))))


(defn eval-expr
  "Tiny simple interpreter."
  [env math-expr]
  (cond
    (sequential? math-expr)
    (if (symbol? (first math-expr))
      (let [fn-name (first math-expr)
            ;;Force errors early
            expr-args (mapv (partial eval-expr env) (rest math-expr))
            operand (get-operand env fn-name)]
        (try
          (apply operand env expr-args)
          (catch Throwable e
            (throw (ex-info (format "Operator %s failed:\n%s" math-expr (.getMessage e))
                            {:math-expression math-expr
                             :error e})))))
      (map partial eval-expr env math-expr))
    (symbol? math-expr)
    (get-operand env math-expr)
    :else
    math-expr))

(defmacro register-math-fn
  [fn-name fn-type-seq]
  (let [red-symbol-remap (get {:/ :div} fn-name fn-name)]
    `(do
       ~(when (fn-type-seq :unary-reduce)
          (let [fn-sym (symbol (str (name red-symbol-remap) "-reduce"))
                sym-name (name fn-sym)]
            `(do
               (defn ~fn-sym
                 ~(format "Reduction with operator %s" (name fn-name))
                 [~'tensor]
                 (unary-reduce ~fn-name ~'tensor))
               (register-symbol! (symbol ~sym-name)
                                           (fn [_# & args#]
                                             (apply ~fn-sym args#))))))
       ~(if (or (fn-type-seq :unary)
                (fn-type-seq :binary))
          `(do
             ~(cond (and (fn-type-seq :unary)
                         (fn-type-seq :binary))
                    `(defn ~(symbol (name fn-name))
                       ~(format "Apply %s in unary or binary context" (name fn-name))
                       ([~'tensor-or-scalar]
                        (unary-op ~fn-name ~'tensor-or-scalar))
                       ([~'tensor-or-scalar & ~'args]
                        (binary-op ~fn-name (concat [~'tensor-or-scalar]
                                                              ~'args))))
                    (fn-type-seq :unary)
                    `(defn ~(symbol (name fn-name))
                       ~(format "Apply unary operator %s" (name fn-name))
                       [~'tensor-or-scalar]
                       (unary-op ~fn-name ~'tensor-or-scalar))
                    (fn-type-seq :binary)
                    `(defn ~(symbol (name fn-name))
                       ~(format "Apply binary operator %s" (name fn-name))
                       [~'tensor-or-scalar & ~'args]
                       (binary-op ~fn-name (concat [~'tensor-or-scalar]
                                                             ~'args))))
             (register-symbol! (symbol ~(name fn-name))
                                         (fn [_# & args#]
                                           (apply ~(symbol (name fn-name)) args#))))))))
