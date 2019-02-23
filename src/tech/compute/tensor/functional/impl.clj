(ns tech.compute.tensor.functional.impl
  "Implementation details of the tensor functional interface"
  (:require [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math.unary-op :as unary-op]
            [tech.compute.cpu.tensor-math.binary-op-impls :as binop-impls]
            [tech.compute.cpu.tensor-math.unary-reduce :as unary-reduce]
            [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.compute.cpu.math-operands :as math-ops
             :refer [as-unary-op as-typed-unary-op
                     as-binary-op as-typed-binary-op]]
            [tech.datatype :as dtype])
  (:import [tech.compute.cpu UnaryOp TypedUnaryOp
            BinaryOp TypedBinaryOp
            UnaryReduce]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defn unary-op
  [op-kwd op-arg]
  (if (number? op-arg)
    (if-let [un-op-base (math-ops/get-unary-operand op-kwd)]
      (if-let [un-op (as-unary-op un-op-base)]
        (.op un-op (double op-arg))
        (let [un-op (as-typed-unary-op un-op-base)]
          (case (dtype/get-datatype op-arg)
            :int8 (.byteOp un-op (unchecked-byte op-arg) 0)
            :int16 (.shortOp un-op (unchecked-short op-arg) 0)
            :int32 (.intOp un-op (unchecked-int op-arg) 0)
            :int64 (.longOp un-op (unchecked-long op-arg) 0)
            :float32 (.floatOp un-op (unchecked-float op-arg) 0)
            :float64 (.doubleOp un-op (unchecked-double op-arg) 0))))
      (throw (ex-info (format "Unable to find scalar %s" op-kwd))))
    (ct/unary-op! (ct/from-prototype op-arg) 1.0 op-arg op-kwd)))


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
  (if-let [untyped-bin-op (math-ops/get-binary-operand op-kwd)]
    (if-let [bin-op (as-binary-op untyped-bin-op)]
      (loop [lhs (unchecked-double (first arglist))
             arglist (rest arglist)]
        (if-let [rhs (first arglist)]
          (recur (.op bin-op (unchecked-double lhs) (unchecked-double rhs))
                 (rest arglist))
          lhs))
      (let [bin-op (as-typed-binary-op untyped-bin-op)]
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


(defn tensor?
  [item]
  (or (ct/acceptable-tensor-buffer? item)
      (ct/tensor? item)))


(defn binary-op
  "Perform a binary operation falling back to the registered interface
  if the operands aren't tensors."
  [op-kwd arglist]
  (when-not (>= (count arglist) 2)
    (throw (ex-info "Binary operations take at least 2 arguments"
                    {:arg-count (count arglist)})))
  (if-let [tens-seq (->> arglist
                         (filter tensor?)
                         seq)]
    (let [accum (ct/assign! (ct/from-prototype (first tens-seq))
                            (first arglist))]
      (doseq [rhs (rest arglist)]
        (ct/binary-op! accum 1.0 accum 1.0 rhs op-kwd))
      accum)
    (scalar-binary-op op-kwd arglist)))


(defn binary-op-fallback
  "Perform a binary operation using provided function if the operands are
  not tensors.  The provided function must take multiple (more than 2) arguments."
  [op-kwd op-scalar-fn arglist]
  (when-not (>= (count arglist) 2)
    (throw (ex-info "")))
  (if-let [tens-seq (->> arglist
                         (filter tensor?)
                         seq)]
    (let [accum (ct/assign! (ct/from-prototype (first tens-seq))
                            (first arglist))]
      (doseq [rhs (rest arglist)]
        (ct/binary-op! accum 1.0 accum 1.0 rhs op-kwd))
      accum)
    (apply op-scalar-fn arglist)))


(defonce ^:dynamic *registered-language-fns* (atom {}))


(defn register-symbol!
  [registration-atom sym-name sym-value]
  (-> (swap! (or registration-atom
                 *registered-language-fns*)
             assoc sym-name sym-value)
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


(defn symbol->str
  [sym]
  (if (namespace sym)
    (str (namespace sym) "/" (name sym))
    (str (name sym))))


(defmacro make-math-fn
  "Make a function that class into the impl for the op type.
  If a scalar fallback is passed in then this is used instead of the tensor
  operator when dealing with things that are not tensors.
  If fn-name is namespaced the function will be registered in the local namespace
  but the system will call the namespaced function as the tensor math op."
  [registration-atom fn-name fn-type-seq & [binary-scalar-fallback]]
  (let [red-symbol-remap (get {:/ :div} fn-name fn-name)
        sym-name (name red-symbol-remap)]
    `(do
       ~(when (fn-type-seq :unary-reduce)
          (let [fn-sym (symbol (str sym-name "-reduce"))
                sym-name (name fn-sym)]
            `(do
               (defn ~(symbol sym-name)
                 ~(format "Reduction with operator %s" sym-name)
                 [~'tensor]
                 (unary-reduce ~fn-name ~'tensor))
               (register-symbol! ~registration-atom
                                 (symbol ~sym-name)
                                 (fn [_# & args#]
                                   (apply ~(symbol sym-name) args#))))))
       ~(if (or (fn-type-seq :unary)
                (fn-type-seq :binary))
          `(do
             ~(cond (and (fn-type-seq :unary)
                         (fn-type-seq :binary))
                    `(defn ~(symbol sym-name)
                       ~(format "Apply %s in unary or binary context" (name fn-name))
                       ([~'tensor-or-scalar]
                        (unary-op ~fn-name ~'tensor-or-scalar))
                       ([~'tensor-or-scalar & ~'args]
                        ~(if binary-scalar-fallback
                           `(binary-op-fallback ~fn-name ~binary-scalar-fallback
                                                (concat [~'tensor-or-scalar]
                                                        ~'args))
                           `(binary-op ~fn-name (concat [~'tensor-or-scalar]
                                                        ~'args)))))
                    (fn-type-seq :unary)
                    `(defn ~(symbol sym-name)
                       ~(format "Apply unary operator %s" (name fn-name))
                       [~'tensor-or-scalar]
                       (unary-op ~fn-name ~'tensor-or-scalar))
                    (fn-type-seq :binary)
                    `(defn ~(symbol sym-name)
                       ~(format "Apply binary operator %s" (name fn-name))
                       [~'tensor-or-scalar & ~'args]
                       ~(if binary-scalar-fallback
                          `(binary-op-fallback ~fn-name
                                               ~binary-scalar-fallback
                                               (concat [~'tensor-or-scalar]
                                                       ~'args))
                          `(binary-op ~fn-name (concat [~'tensor-or-scalar]
                                                       ~'args)))))
             (register-symbol! ~registration-atom
                               (symbol ~sym-name)
                               (fn [_# & args#]
                                 (apply ~(symbol sym-name) args#))))))))
