(ns tech.compute.tensor.functional
  "Functional math for tensor system.  Defines both an AST and live implementation.
  Most of the functionality is based on the registered symbols in the cpu context;
  other contexts will almost certainly *not* support as many symbols."
  (:require [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.compute.tensor.functional.impl :as func-impl])
  (:refer-clojure :exclude [+ - / *
                            <= < >= >
                            min max
                            bit-xor bit-and bit-and-not bit-not bit-set bit-test
                            bit-or bit-flip bit-clear
                            bit-shift-left bit-shift-right unsigned-bit-shift-right
                            quot rem]))


(def ^:private operator-map
  (let [{:keys [unary binary unary-reduce]} (cpu-tm/all-known-operators)]
    (->> (concat (map vector (repeat :unary) unary)
                 (map vector (repeat :binary) binary)
                 (map vector (repeat :unary-reduce) unary-reduce))
         (group-by second)
         (map (fn [[k val-seq]]
                [k (->> (map first val-seq)
                        set)]))
         (into {}))))


(defmacro ^:private register-math-fn
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
                 (func-impl/unary-reduce ~fn-name ~'tensor))
               (func-impl/register-symbol! (symbol ~sym-name)
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
                        (func-impl/unary-op ~fn-name ~'tensor-or-scalar))
                       ([~'tensor-or-scalar & ~'args]
                        (func-impl/binary-op ~fn-name (concat [~'tensor-or-scalar]
                                                              ~'args))))
                    (fn-type-seq :unary)
                    `(defn ~(symbol (name fn-name))
                       ~(format "Apply unary operator %s" (name fn-name))
                       [~'tensor-or-scalar]
                       (func-impl/unary-op ~fn-name ~'tensor-or-scalar))
                    (fn-type-seq :binary)
                    `(defn ~(symbol (name fn-name))
                       ~(format "Apply binary operator %s" (name fn-name))
                       [~'tensor-or-scalar & ~'args]
                       (func-impl/binary-op ~fn-name (concat [~'tensor-or-scalar]
                                                             ~'args))))
             (func-impl/register-symbol! (symbol ~(name fn-name))
                                         (fn [_# & args#]
                                           (apply ~(symbol (name fn-name)) args#))))))))


(defmacro ^:private make-standard-fns
  []
  `(do
     ~@(for [[op-name op-typeset] operator-map]
         `(register-math-fn ~op-name ~op-typeset))))


(make-standard-fns)
