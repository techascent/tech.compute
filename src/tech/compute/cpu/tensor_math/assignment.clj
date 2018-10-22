(ns tech.compute.cpu.tensor-math.assignment
  (:require [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math.nio-access
             :refer [b-put b-get datatype-iterator
                     store-datatype-cast-fn
                     read-datatype-cast-fn
                     item->typed-nio-buffer
                     ] :as nio-access]
            [tech.compute.cpu.tensor-math.addressing
             :refer [get-elem-dims->address
                     max-shape-from-dimensions]]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.datatype.java-primitive :as primitive]))


(defmacro ^:private assign-constant-impl
  [datatype]
  `(fn [buffer# dimensions# value# n-elems#]
     (let [n-elems# (long n-elems#)
           buffer# (item->typed-nio-buffer ~datatype buffer#)
           idx->address# (get-elem-dims->address dimensions# (get dimensions# :shape))
           value# (unsigned/datatype->jvm-cast-fn :ignored ~datatype value#)]
       (parallel/parallel-for
        idx# n-elems#
        (b-put buffer#
              (.idx_to_address idx->address# idx#)
              value#)))))


(def assign-constant-map
  (->> (datatype-iterator assign-constant-impl)
       (into {})))


(defmacro ^:private marshalling-assign-fn
  [lhs-dtype rhs-dtype]
  `(fn [dest# dest-dim#
        src# src-dim#
        n-elems#]
     (let [dest# (item->typed-nio-buffer ~lhs-dtype dest#)
           src# (item->typed-nio-buffer ~rhs-dtype src#)
           max-shape# (max-shape-from-dimensions dest-dim# src-dim#)
           dest-idx->address# (get-elem-dims->address dest-dim# max-shape#)
           src-idx->address# (get-elem-dims->address src-dim# max-shape#)
           n-elems# (long n-elems#)]
       (parallel/parallel-for
        idx# n-elems#
        (b-put dest# (.idx_to_address dest-idx->address# idx#)
               (store-datatype-cast-fn
                ~lhs-dtype
                (read-datatype-cast-fn
                 ~rhs-dtype
                 (b-get src# (.idx_to_address src-idx->address# idx#)))))))))


(defmacro ^:private generate-all-marshalling-assign-fns
  []
  (->> unsigned/all-possible-datatype-pairs
       (map (fn [[lhs-dtype rhs-dtype]]
              [[lhs-dtype rhs-dtype]
               `(marshalling-assign-fn ~lhs-dtype ~rhs-dtype)]))
       (into {})))


(def assign!-map
  (generate-all-marshalling-assign-fns))
