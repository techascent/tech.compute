(ns tech.compute.tensor.functional
  "Functional math for tensor system.  Defines both an AST and live implementation.
  Most of the functionality is based on the registered symbols in the cpu context;
  other contexts will almost certainly *not* support as many symbols."
  (:require [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.datatype :as dtype]
            [tech.compute.tensor.functional.impl :as func-impl])
  (:refer-clojure :exclude [+ - / *
                            <= < >= >
                            min max
                            bit-xor bit-and bit-and-not bit-not bit-set bit-test
                            bit-or bit-flip bit-clear
                            bit-shift-left bit-shift-right unsigned-bit-shift-right
                            quot rem cast not and or]))


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


(def scalar-override-fn-map
  {:> clojure.core/>
   :>= clojure.core/>=
   :< clojure.core/<
   :<= clojure.core/<=
   :eq #(clojure.core/= %1 (dtype/cast %2 (dtype/get-datatype %1)))
   :not-eq #(clojure.core/not= %1 (dtype/cast %2 (dtype/get-datatype %1)))})


(defmacro ^:private make-standard-fns
  []
  `(do
     ~@(for [[op-name op-typeset] operator-map]
         (if-let [override (get scalar-override-fn-map op-name)]
           `(func-impl/make-math-fn nil ~op-name ~op-typeset ~override)
           `(func-impl/make-math-fn nil ~op-name ~op-typeset)))))


(make-standard-fns)


(defn cast
  "Cast an item to a given datatype"
  [item dtype]
  (if (func-impl/tensor? item)
    (ct/clone item :datatype dtype)
    (dtype/cast item dtype)))

(func-impl/register-symbol! nil 'cast (fn [_ & args]
                                        (apply cast args)))
