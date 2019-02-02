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
            [tech.datatype.java-primitive :as primitive]
            [tech.compute.cpu.tensor-math.writers :as writers]))


(defmacro ^:private assign-constant-impl
  [datatype]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype datatype)]
    `(fn [buffer# dimensions# value# n-elems#]
       (let [value# (unsigned/datatype->jvm-cast-fn :ignored ~datatype value#)
             writer# (writers/get-parallel-writer ~jvm-datatype)
             max-shape# (max-shape-from-dimensions dimensions#)]
         (writer# buffer# dimensions# max-shape# n-elems#
                  (primitive/make-converter ~jvm-datatype value#))))))


(def assign-constant-map
  (->> (datatype-iterator assign-constant-impl)
       (into {})))


(defmacro ^:private marshalling-assign-fn
  [lhs-dtype rhs-dtype]
  (let [jvm-datatype (unsigned/datatype->jvm-datatype lhs-dtype)]
    `(fn [dest# dest-dim#
          src# src-dim#
          n-elems#]
       (let [src# (item->typed-nio-buffer ~rhs-dtype src#)
             max-shape# (max-shape-from-dimensions dest-dim# src-dim#)
             src-idx->address# (get-elem-dims->address src-dim# max-shape#)
             writer# (writers/get-parallel-writer ~jvm-datatype)]
         (writer# dest# dest-dim# max-shape# n-elems#
                  (primitive/make-converter
                   ~jvm-datatype
                   (store-datatype-cast-fn
                    ~lhs-dtype
                    (read-datatype-cast-fn
                     ~rhs-dtype
                     (b-get src# (.idx_to_address src-idx->address# ~'idx))))))))))


(defmacro ^:private generate-all-marshalling-assign-fns
  []
  (->> unsigned/all-possible-datatype-pairs
       (map (fn [[lhs-dtype rhs-dtype]]
              [[lhs-dtype rhs-dtype]
               `(marshalling-assign-fn ~lhs-dtype ~rhs-dtype)]))
       (into {})))


(def assign!-map
  (generate-all-marshalling-assign-fns))
