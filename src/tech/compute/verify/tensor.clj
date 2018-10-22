(ns tech.compute.verify.tensor
  (:require [tech.compute :as compute]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.dimensions :as ct-dims]
            [tech.compute.driver :as drv]
            [tech.datatype :as dtype]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as stats]
            [tech.resource :as resource]
            [tech.compute.verify.utils :as verify-utils]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.compute.tensor.defaults :as defaults]))


(defmacro tensor-context
  [stream datatype & body]
  `(resource/with-resource-context
     (defaults/tensor-context
      ~stream
      ~datatype
       ~@body)))


(defmacro tensor-default-context
  [driver datatype & body]
  `(resource/with-resource-context
     (defaults/tensor-driver-context
      ~driver
      ~datatype
      ~@body)))


(defn assign-constant!
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [tensor (ct/->tensor (partition 3 (range 9)))]
     (is (= (ct/ecount tensor) 9))
     (is (m/equals (range 9)
                   (ct/to-float-array tensor)))
     (ct/assign! tensor 1)
     (is (m/equals (repeat 9 1)
                   (ct/to-float-array tensor)))
     (let [rows (ct/rows tensor)
           columns (ct/columns tensor)]
       (doseq [row rows]
         (ct/assign! row 2))
       (is (m/equals (repeat 9 2)
                     (ct/to-float-array tensor)))
       (let [[c1 c2 c3] columns]
         (ct/assign! c1 1)
         (ct/assign! c2 2)
         (ct/assign! c3 3))
       (is (m/equals (flatten (repeat 3 [1 2 3]))
                     (ct/to-float-array tensor)))))))


(defn assign-marshal
  "Assignment must be capable of marshalling data.  This is an somewhat difficult challenge
for the cuda backend."
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [tensor (ct/->tensor (partition 3 (range 9)))
         intermediate (ct/new-tensor [3 3] :datatype :int32)
         final (ct/new-tensor [3 3] :datatype :float32)]
     (ct/assign! intermediate tensor)
     (ct/assign! final intermediate)
     (is (m/equals (range 9)
                   (ct/to-float-array final)))
     (let [bcast-tensor (ct/->tensor [5 6 7])
           dest-tensor (ct/new-tensor [3 3])]
       (ct/assign! dest-tensor bcast-tensor)
       (is (m/equals (flatten (repeat 3 [5 6 7]))
                     (ct/to-float-array dest-tensor)))))))


(defn unary-op
  [driver datatype]
  (tensor-default-context driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 1)))]
     (ct/unary-op! tens-b 2.5 tens-a :ceil)
     (is (m/equals (mapv #(Math/ceil (* ^double % (dtype/cast 2.5 datatype))) (range 9))
                   (ct/to-double-array tens-b)))
     (ct/unary-op! tens-b 1.0 tens-b :-)
     (when-not (unsigned/unsigned-datatype? datatype)
       (is (m/equals (mapv #(- (Math/ceil (* ^double % (dtype/cast 2.5 datatype))))
                           (range 9))
                     (ct/to-double-array tens-b))))

     (let [src-data [0 1 2 3 4]
           tens-src (ct/->tensor src-data)
           tens-b (ct/->tensor src-data)]
       (ct/unary-op! tens-b 1.0 tens-src :exp)
       (is (m/equals (mapv #(dtype/cast (Math/exp (double %)) datatype)
                           src-data)
                     (ct/to-double-array tens-b)))
       (ct/unary-op! tens-b 1.0 tens-src :sqrt)
       (is (m/equals (mapv #(dtype/cast (Math/sqrt (double %)) datatype)
                           src-data)
                     (ct/to-double-array tens-b)))))))


(defn channel-op
  [driver datatype]
  (tensor-default-context driver datatype
   (let [tens-a (ct/->tensor (partition 3 (partition 3 (range 18))))
         tens-chan (ct/in-place-reshape (ct/->tensor (range 3)) [3 1])
         tens-result (ct/new-tensor [2 3 3])]

     (ct/binary-op! tens-result 1.0 tens-a 1.0 tens-chan :*)
     (is (m/equals [0.0, 0.0, 0.0, 3.0, 4.0, 5.0,
                    12.0, 14.0, 16.0, 0.0, 0.0, 0.0, 12.0,
                    13.0, 14.0, 30.0, 32.0, 34.0]
                   (ct/to-double-array tens-result))))))


(defn binary-constant-op
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 1)))]

     (when-not (= datatype :int8)
       (ct/binary-op! tens-b 2.0 tens-a 3.0 4.0 :*)
       (is (m/equals (mapv #(* 24 %) (range 9))
                     (ct/to-double-array tens-b))))

     (ct/binary-op! tens-b 2.0 tens-a 3.0 4.0 :+)
     (is (m/equals (mapv #(+ 12 (* 2 %)) (range 9))
                   (ct/to-double-array tens-b)))


     ;;Check reversing operands works.
     (ct/binary-op! tens-b 3.0 4.0 2.0 tens-a :-)
     (is (m/equals (mapv #(- 12 (* 2 %)) (range 9))
                   (ct/to-double-array tens-b)))

     (ct/assign! tens-b 1.0)
     (is (m/equals (repeat 9 1)
                   (ct/to-double-array tens-b)))

     ;;Check accumulation
     (ct/binary-op! tens-b 1.0 tens-b 1.0 1.0 :+)
     (is (m/equals (mapv #(+ 1 %) (repeat 9 1))
                   (ct/to-double-array tens-b))))))


(defn binary-op
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 2)))
         tens-c (ct/->tensor (partition 3 (repeat 9 10)))]
     (ct/binary-op! tens-c 2.0 tens-a 2.0 tens-b :*)
     (is (m/equals (mapv #(* 2 2 2 %) (flatten (partition 3 (range 9))))
                   (ct/to-double-array tens-c)))
     (ct/binary-op! tens-b 1.0 tens-c 2.0 tens-a :-)
     (is (m/equals [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
                   (ct/to-double-array tens-b)))

     ;;A binary accumulation operation where the destination is the same
     ;;as one of the operands.
     (ct/binary-op! tens-c 1.0 tens-c 2.0 tens-a :-)
     (is (m/equals [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
                   (ct/to-double-array tens-c)))
     (let [tens-c-small (ct/subvector tens-c 0 :length 3)
           sub-fn (fn [nth-idx]
                    (->> (partition 3 (range 9))
                         (map #(nth % nth-idx))))
           c-data (vec (ct/to-double-array tens-c))]
       (ct/binary-op! tens-c-small 1.0 tens-c-small 2.0 tens-a :-)
       (is (m/equals (mapv (fn [elem-idx]
                             (apply -
                                    (nth c-data elem-idx)
                                    (mapv #(* 2.0 %)
                                          (sub-fn elem-idx))))
                           [0 1 2])
                     (ct/to-double-array tens-c-small)))
       (let [c-data (vec (ct/to-double-array tens-c-small))]
         (ct/binary-op! tens-c-small 2.0 tens-a 1.0 tens-c-small :+)
         (is (m/equals (mapv (fn [elem-idx]
                               (reduce (fn [result a-elem]
                                         (+ a-elem result))
                                       (nth c-data elem-idx)
                                       (map #(* 2.0 %) (sub-fn elem-idx))))
                             [0 1 2])
                       (ct/to-double-array tens-c-small)))))
     (when (contains? #{:float32 :float64} datatype)
       (let [n-batches 3
             n-channels 5
             img-dim 4
             big-pools (repeatedly (*  n-channels n-batches)
                                   (fn []
                                     (vec (repeatedly (* img-dim img-dim) rand))))
             sums (->> (mapv #(apply + %) big-pools)
                       (partition n-channels)
                       (apply m/add))

             big-m (ct/->tensor (->> (flatten big-pools)
                                     (partition img-dim)
                                     (partition img-dim)
                                     (partition n-channels)))
             test-vec (-> (ct/new-tensor [n-channels])
                          (ct/in-place-reshape [n-channels 1 1]))]
         ;;broadcasting summation
         (ct/binary-op! test-vec 1.0 test-vec 1.0 big-m :+)
         (is (m/equals sums
                       (ct/to-double-array test-vec)
                       1e-4))))

     (let [tens-a (ct/->tensor (repeat 4 (partition 3 (range 9))))
           result (ct/new-tensor (m/shape tens-a))]
       (ct/binary-op! result 1.0 tens-a 1.0 5 :eq)
       (is (m/equals (mapv #(if (= (long %) 5)
                              1
                              0)
                           (ct/to-double-array tens-a))
                     (ct/to-double-array result))))

     (let [tens-a (ct/new-tensor [4 3 3])
           bias (-> (ct/->tensor [1 2 3 4])
                    (ct/in-place-reshape [4 1 1]))]
       (ct/binary-op! tens-a 1.0 tens-a 1.0 bias :+)
       (is (m/equals (flatten
                      (map #(repeat 9 %) [1 2 3 4]))
                     (ct/to-double-array tens-a))))
     ;;bias-gradient calculation
     (let [tens-a (ct/->tensor  (mapv #(->> (repeat 9 %)
                                            (partition 3)) [1 2 3 4]))
           bias (-> (ct/new-tensor [4])
                    (ct/in-place-reshape [4 1 1]))]
       (ct/binary-op! bias 1.0 tens-a 1.0 bias :+)
       (is (m/equals [9 18 27 36]
                     (ct/to-double-array bias))))

     (let [tens-1 (ct/->tensor [1 1 0 0])
           tens-2 (ct/->tensor [-1 1 2 -2])
           result (ct/new-tensor (m/shape tens-2))]

        ;;; > greater than
       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :>)
       (is (m/equals [1 0 0 1]
                     (ct/to-double-array result)))

       ;;; >= greater than or equal to
       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :>=)
       (is (m/equals [1 1 0 1]
                     (ct/to-double-array result)))

       ;;; < less than
       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :<)
       (is (m/equals [0 0 1 0]
                     (ct/to-double-array result)))

       ;;; <= less than or equal to
       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :<=)
       (is (m/equals [0 1 1 0]
                     (ct/to-double-array result)))))

   ;; bit-xor
   (let [tens-1 (ct/->tensor [1 1 0 0])
         tens-2 (ct/->tensor [1 1 1 1])
         result (ct/new-tensor (m/shape tens-2))]

       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :bit-xor)
       (is (m/equals [0 0 1 1]
                     (ct/to-double-array result))))))


(defn gemm
  ;;Test gemm, also test that submatrixes are working and defined correctly.
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 2)))
         tens-c (ct/->tensor (partition 3 (repeat 9 10)))]
     (ct/gemm! tens-c false false 1 tens-a tens-b 0)
     (is (m/equals (ct/to-double-array tens-c)
                   [6.0 6.0 6.0 24.0 24.0 24.0 42.0 42.0 42.0]))
     (ct/gemm! tens-c true false 1 tens-a tens-b 0)
     (is (m/equals (ct/to-double-array tens-c)
                   [18.0, 18.0, 18.0, 24.0, 24.0, 24.0, 30.0, 30.0, 30.0]))
     (let [tens-a (ct/submatrix tens-a 0 2 0 2)
           tens-b (ct/submatrix tens-b 0 2 0 2)
           tens-c-sub (ct/submatrix tens-c 0 2 0 2)]
       (ct/gemm! tens-c-sub false false 1 tens-a tens-b 0)
       (is (m/equals (ct/to-double-array tens-c-sub)
                     [2 2 14 14]))
       (is (m/equals (ct/to-double-array tens-c)
                     [2.0, 2.0, 18.0, 14.0, 14.0, 24.0, 30.0, 30.0, 30.0])))

     (try
       (ct/assign! tens-c 10)
       (ct/gemm! tens-c false false 1 tens-a tens-b 1)
       (is (m/equals (ct/to-double-array tens-c)
                     [16.0 16.0 16.0 34.0 34.0 34.0 52.0 52.0 52.0]))

       (let [tens-a (ct/submatrix tens-a 1 2 1 2)
             tens-b (ct/submatrix tens-b 0 2 0 2)
             tens-c-sub (ct/submatrix tens-c 1 2 1 2)]
         (ct/gemm! tens-c-sub false false 1 tens-a tens-b 1)
         (is (m/equals [52.0, 52.0, 82.0, 82.0]
                       (ct/to-double-array tens-c-sub)))
         (is (m/equals [16.0, 16.0, 16.0, 34.0, 52.0, 52.0, 52.0, 82.0, 82.0]
                       (ct/to-double-array tens-c))))
       ;;Exception here is fine.  The gemm operation doesn't always support
       ;;summing into C
       (catch Throwable e
         nil)))))


(defn ternary-op-select
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [dest (ct/->tensor (repeat 10 0))
         x-arg (ct/->tensor (range -5 5))
         y-arg (ct/->tensor (range 10))
         z-arg (ct/->tensor (repeat 10 2))]
     (ct/ternary-op! dest 1 x-arg 2.0 y-arg 3.0 z-arg :select)
     (is (m/equals [0 2 4 6 8 6 6 6 6 6]
                   (ct/to-double-array dest)))
     (ct/ternary-op! dest 1 x-arg 1.0 -1 3.0 z-arg :select)
     (is (m/equals [-1 -1 -1 -1 -1 6 6 6 6 6]
                   (ct/to-double-array dest)))
     (ct/ternary-op! dest 1 x-arg 3.0 z-arg 1.0 -1 :select)
     (is (m/equals [6 6 6 6 6 -1 -1 -1 -1 -1]
                   (ct/to-double-array dest)))
     (ct/ternary-op! dest 1 x-arg 3.0 2.0 1.0 -1 :select)
     (is (m/equals [6 6 6 6 6 -1 -1 -1 -1 -1]
                   (ct/to-double-array dest)))
     (ct/ternary-op! dest 1 x-arg 1.0 -1 3.0 2.0 :select)
     (is (m/equals [-1 -1 -1 -1 -1 6 6 6 6 6]
                   (ct/to-double-array dest))))))


(defn unary-reduce
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [dest (ct/new-tensor [10 1])
         src-data [0 3 5 2 1 9 5 7 7 2]
         src (ct/->tensor (repeat 10 src-data))]
     (ct/unary-reduce! dest 2.0 src :max)
     (is (m/equals (repeat 10 18)
                   (ct/to-double-array dest)))
     (ct/unary-reduce! dest 1.0 src :sum)
     (is (m/equals (repeat 10 (apply + src-data))
                   (ct/to-double-array dest)))
     (ct/unary-reduce! dest 1.0 src :mean)
     (is (m/equals (repeat 10 (dtype/cast (/ (apply + src-data)
                                             (count src-data))
                                          datatype))
                   (ct/to-double-array dest))))))


(defn transpose
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [img-dim 4
         img-tensor (ct/->tensor
                     (->> (repeat (* img-dim img-dim) [1 2 3])
                          (partition img-dim)))
         planar-tensor (ct/transpose img-tensor [2 0 1])
         rgb-tensor (ct/transpose planar-tensor [1 2 0])]
     (is (m/equals (flatten (concat (repeat (* img-dim img-dim) 1)
                                    (repeat (* img-dim img-dim) 2)
                                    (repeat (* img-dim img-dim) 3)))
                   (ct/to-double-array planar-tensor)))

     (is (m/equals (flatten (repeat (* img-dim img-dim) [1 2 3]))
                   (ct/to-double-array rgb-tensor))))))


(defn mask
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [r-pix (int 1)
         g-pix (int 2)
         b-pix (int 3)
         ;;Load a single image to r,g,b planes
         rgba (dtype/unchecked-cast
               (+ r-pix
                  (bit-shift-left g-pix 8)
                  (bit-shift-left b-pix 16)
                  (bit-shift-left (int 255) 24))
               datatype)
         img-dim 4
         img-tensor (ct/->tensor
                     (->> (repeat (* img-dim img-dim) rgba)
                          (partition img-dim)))
         mask-tensor (-> (ct/->tensor [0xFF
                                       (bit-shift-left 0xFF 8)
                                       (bit-shift-left 0xFF 16)])
                         (ct/in-place-reshape [3 1 1]))
         div-tensor (-> (ct/->tensor [1
                                      (bit-shift-left 1 8)
                                      (bit-shift-left 1 16)])
                        (ct/in-place-reshape [3 1 1]))
         result (ct/new-tensor [3 img-dim img-dim])]
     (ct/binary-op! result 1.0 img-tensor 1.0 mask-tensor :bit-and)
     (ct/binary-op! result 1.0 result 1.0 div-tensor :/)
     (is (m/equals (flatten (concat (repeat (* img-dim img-dim) 1)
                                    (repeat (* img-dim img-dim) 2)
                                    (repeat (* img-dim img-dim) 3)))
                   (ct/to-double-array result))))))


(defn select
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [mat-tens (ct/->tensor (repeat 2 (partition 3 (range 9))))]
     (let [sel-tens (ct/select mat-tens :all :all [1 2])]
       (is (m/equals (flatten (repeat 2 [1 2 4 5 7 8]))
                     (ct/to-double-array sel-tens)))
       (is (m/equals [2 3 2]
                     (m/shape sel-tens))))
     (let [sel-tens (ct/select mat-tens :all :all [2])]
         (is (m/equals (flatten (repeat 2 [2 5 8]))
                       (ct/to-double-array sel-tens)))
         (is (m/equals [2 3 1]
                       (m/shape sel-tens))))
     (let [sel-tens (ct/select mat-tens :all :all 2)]
         (is (m/equals (flatten (repeat 2 [2 5 8]))
                       (ct/to-double-array sel-tens)))
         (is (m/equals [2 3]
                       (m/shape sel-tens)))
         (is (not (ct/dense? sel-tens))))
     (let [sel-tens (ct/select mat-tens :all [1 2] :all)]
         (is (m/equals (flatten (repeat 2 [3 4 5 6 7 8]))
                       (ct/to-double-array sel-tens)))
         (is (m/equals [2 2 3]
                       (m/shape sel-tens)))
         (is (not (ct/dense? sel-tens))))
     (let [sel-tens (ct/select mat-tens :all [2] :all)]
         (is (m/equals (flatten (repeat 2 [6 7 8]))
                       (ct/to-double-array sel-tens)))
         (is (m/equals [2 1 3]
                       (m/shape sel-tens)))
         (is (not (ct/dense? sel-tens))))
     (let [sel-tens (ct/select mat-tens :all 0 :all)]
       (is (m/equals (flatten (repeat 2 [0 1 2]))
                     (ct/to-double-array sel-tens)))
       (is (m/equals [2 3]
                     (m/shape sel-tens)))
       (is (not (ct/dense? sel-tens))))

     (let [sel-tens (ct/select mat-tens [1] [1] :all)]
       (is (m/equals [3 4 5]
                     (ct/to-double-array sel-tens)))
       (is (m/equals [1 1 3]
                     (m/shape sel-tens)))
       (is (ct/dense? sel-tens)))

     (let [sel-tens (ct/select mat-tens 1 1 :all)]
       (is (m/equals [3 4 5]
                     (ct/to-double-array sel-tens)))
       (is (m/equals [3]
                     (m/shape sel-tens)))
       (is (ct/dense? sel-tens))
       (is (ct/as-vector sel-tens)))

     (let [sel-tens (ct/select mat-tens 1 :all 2)]
       (is (m/equals [2 5 8]
                     (ct/to-double-array sel-tens)))
       (is (m/equals [3]
                     (m/shape sel-tens)))
       (is (not (ct/dense? sel-tens)))))))


(defn select-with-persistent-vectors
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [mat-tens (ct/->tensor (repeat 2 (partition 3 (range 9))))
         channels-first (ct/transpose mat-tens [2 0 1])
         bgr (ct/select channels-first [2 1 0] :all :all)]
     (is (= [[[2 5 8]
              [2 5 8]]
             [[1 4 7]
              [1 4 7]]
             [[0 3 6]
              [0 3 6]]]
            (ct/to-jvm bgr :datatype :int32))))))


(defn select-transpose-interaction
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [img-dim 4
         mat-tens (ct/->tensor (partition img-dim (repeat (* img-dim img-dim) [1 2 3])))
         planar-tens (ct/transpose mat-tens [2 0 1])
         n-pixels (* img-dim img-dim)]
     (let [r-tens (ct/select planar-tens 0 :all :all)
           g-tens (ct/select planar-tens 1 :all :all)
           b-tens (ct/select planar-tens 2 :all :all)]
       (is (m/equals (repeat n-pixels 1) (ct/to-double-array r-tens)))
       (is (m/equals (repeat n-pixels 2) (ct/to-double-array g-tens)))
       (is (m/equals (repeat n-pixels 3) (ct/to-double-array b-tens)))
       (let [bgr-tens (ct/new-tensor [img-dim img-dim 3])
             bgr-planes (ct/transpose bgr-tens [2 0 1])]
         (m/assign! (ct/select bgr-planes 0 :all :all) b-tens)
         (m/assign! (ct/select bgr-planes 1 :all :all) g-tens)
         (m/assign! (ct/select bgr-planes 2 :all :all) r-tens)
         (is (m/equals (flatten (partition img-dim (repeat (* img-dim img-dim) [3 2 1])))
                       (ct/to-double-array bgr-tens))))))))




(defn rand-operator
  [driver datatype]
  (tensor-default-context
   driver datatype
   (testing "Gaussian rand"
     (let [test-vec (->> (range 1 11)
                         (mapcat (fn [idx]
                                   (let [tens (ct/rand! (ct/new-tensor [100000])
                                                            (ct/gaussian-distribution
                                                             :mean idx :variance idx))
                                         values (ct/to-double-array tens)]
                                     [(stats/mean values)
                                      (stats/variance values)])))
                         vec)]
       (is (m/equals [1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10]
                     test-vec
                     1))))
   (testing "Flat rand"
     (let [test-vec (->> (range 1 11)
                         (mapcat (fn [idx]
                                   (let [tens (ct/rand! (ct/new-tensor [10000])
                                                            (ct/flat-distribution
                                                             :minimum (- idx 2)
                                                             :maximum (+ idx 2)))
                                         values (ct/to-double-array tens)]
                                     [(stats/mean values)])))
                         vec)]
       (is (m/equals [1 2 3 4 5 6 7 8 9 10]
                     test-vec
                     1))))))



(defn indexed-tensor
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [mat-tens (ct/->tensor (repeat 2 (partition 3 (range 9))))
         ;;Indexing only guaranteed to work for integers
         index-tens (ct/->tensor [1 2 1 2 1 2] :datatype :int32)
         sel-tens (ct/select mat-tens :all :all index-tens)]
     (testing "Basic assigning and sanity"
       (is (not (ct-dims/access-increasing? (:dimensions sel-tens))))
       (is (= [2 3 6]
              (ct/shape sel-tens)))
       (is (m/equals (flatten (repeat 2 [1 2 1 2 1 2 4 5 4 5 4 5 7 8 7 8 7 8]))
                     (vec (ct/to-double-array sel-tens)))))

     ;;Setup a test for what we do in centerloss
     (testing "Center loss use case"
       (let [centers (ct/select mat-tens 0 (ct/->tensor [0 1 0 2 1] :datatype :int32) :all)
             batch-data (ct/new-tensor [5 3])]
         (ct/assign! batch-data centers)
         (is (m/equals [0 1 2 3 4 5 0 1 2 6 7 8 3 4 5]
                       (vec (ct/to-double-array batch-data))))
         ;;Because we are using indexing, we *have* to use the actual compare-and-set versions of things.
         ;;This test should highlight that by producing inconsistent results if the backend doesn't realize
         ;;this situation.
         (ct/binary-op! centers 1.0 batch-data 1.0 centers :+)
         (is (m/equals [0 3 6 9 12 15 0 3 6 12 14 16 9 12 15]
                       (vec (ct/to-double-array centers))))))
     (testing "Broadcasting, multiple indexes in one tensor"
       (let [mat-tens (ct/->tensor (->> (range 9)
                                        (partition 3)))
             sel-tens (ct/select mat-tens
                                 (ct/->tensor [0 1 1 2] :datatype :int32)
                                 (ct/->tensor [2 1 0] :datatype :int32))
             dest-tens (ct/new-tensor [8 6])]
         (ct/assign! dest-tens sel-tens)
         (is (m/equals (vec (flatten [[2.0 1.0 0.0]
                                      [2.0 1.0 0.0]
                                      [5.0 4.0 3.0]
                                      [5.0 4.0 3.0]
                                      [5.0 4.0 3.0]
                                      [5.0 4.0 3.0]
                                      [8.0 7.0 6.0]
                                      [8.0 7.0 6.0]
                                      [2.0 1.0 0.0]
                                      [2.0 1.0 0.0]
                                      [5.0 4.0 3.0]
                                      [5.0 4.0 3.0]
                                      [5.0 4.0 3.0]
                                      [5.0 4.0 3.0]
                                      [8.0 7.0 6.0]
                                      [8.0 7.0 6.0]]))
                       (vec (ct/to-double-array dest-tens)))))))))


(defn magnitude-and-mag-squared
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [num-elems 10
         num-rows 10
         src-data (partition num-elems (range (* num-rows num-elems)))
         src-tensor (ct/->tensor src-data)
         dst-tensor (ct/new-tensor [num-rows 1])]
     (ct/unary-reduce! dst-tensor 1.0 src-tensor :magnitude-squared)
     (is (m/equals (->> (flatten src-data)
                        (map #(* % %))
                        (partition num-rows)
                        (mapv #(reduce + %))
                        vec)
                   (vec (ct/to-double-array dst-tensor))
                   1e-4))

     (ct/unary-reduce! dst-tensor 1.0 src-tensor :magnitude)
     (is (m/equals (->> (flatten src-data)
                        (map #(* % %))
                        (partition num-rows)
                        (mapv (comp #(Math/sqrt (double %))
                                    #(reduce + %)))
                        vec)
                   (vec (ct/to-double-array dst-tensor))
                   1e-4)))))


(defn constrain-inside-hypersphere
  [driver datatype]
  (tensor-default-context
   driver datatype
   (let [num-elems 10
         num-rows 10
         src-data (partition num-elems (range (* num-rows num-elems)))
         radius 100
         src-tensor (ct/->tensor src-data)
         mul-tensor (ct/new-tensor [(first (ct/shape src-tensor)) 1])]
     (ct/constrain-inside-hypersphere! src-tensor mul-tensor radius)
     (let [magnitudes (mapv m/magnitude
                            (partition num-rows
                                       (ct/to-double-array src-tensor)))]
       (is (every? #(<= (double %) (+ radius 1e4))
                   (map m/magnitude
                        (partition num-rows
                                   (ct/to-double-array src-tensor)))))))))
