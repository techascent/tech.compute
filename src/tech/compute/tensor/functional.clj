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


(defmacro ^:private make-standard-fns
  []
  `(do
     ~@(for [[op-name op-typeset] operator-map]
         `(func-impl/register-math-fn ~op-name ~op-typeset))))


(make-standard-fns)
