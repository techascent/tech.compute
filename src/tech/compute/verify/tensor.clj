(ns tech.compute.verify.tensor
  (:require [tech.compute.context :as compute-ctx]
            [tech.v2.tensor.impl :as dtt-impl]
            [tech.v2.datatype.functional :as dfn]
            [tech.v2.datatype :as dtype]
            [tech.v2.tensor :as dtt]
            [tech.compute.tensor :as ct]
            [tech.resource :as resource]
            [clojure.test :refer :all]))


(defmacro verify-context
  [driver datatype & body]
  `(resource/stack-resource-context
    (compute-ctx/with-context
      {:driver ~driver}
      (dtt-impl/with-datatype
        ~datatype
        ~@body))))


(defn assign-constant!
  [driver datatype]
  (verify-context
   driver datatype
   (let [tensor (ct/->tensor (partition 3 (range 9)))]
     (is (= (dtype/ecount tensor) 9))
     (is (dfn/equals (range 9)
                   (ct/->float-array tensor)))
     (ct/assign! tensor 1)
     (is (dfn/equals (repeat 9 1)
                   (ct/->float-array tensor)))
     (let [rows (ct/rows tensor)
           columns (ct/columns tensor)]
       (doseq [row rows]
         (ct/assign! row 2))
       (is (dfn/equals (repeat 9 2)
                     (ct/->float-array tensor)))
       (let [[c1 c2 c3] columns]
         (ct/assign! c1 1)
         (ct/assign! c2 2)
         (ct/assign! c3 3))
       (is (dfn/equals (flatten (repeat 3 [1 2 3]))
                     (ct/->float-array tensor)))))))


(defn assign-marshal
  "Assignment must be capable of marshalling data.  This is an somewhat difficult
  challenge for the cuda backend."
  [driver datatype]
  (verify-context
   driver datatype
   (let [tensor (ct/->tensor (partition 3 (range 9)))
         intermediate (ct/new-tensor [3 3] {:datatype :int32})
         final (ct/new-tensor [3 3] {:datatype :float32})]
     (ct/assign! intermediate tensor)
     (ct/assign! final intermediate)
     (is (dfn/equals (range 9)
                   (ct/->float-array final)))
     (let [bcast-tensor (ct/->tensor [5 6 7])
           dest-tensor (ct/new-tensor [3 3])]
       (ct/assign! dest-tensor bcast-tensor)
       (is (dfn/equals (flatten (repeat 3 [5 6 7]))
                     (ct/->float-array dest-tensor)))))))
