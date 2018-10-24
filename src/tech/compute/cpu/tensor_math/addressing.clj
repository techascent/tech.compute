(ns tech.compute.cpu.tensor-math.addressing
  (:require  [tech.compute.tensor.dimensions :as ct-dims]
             [tech.compute.tensor :as ct]
             [tech.compute.tensor.utils :as ct-utils]))


(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


;;Need the interface to get correct type hinting to avoid boxing/unboxing every index.
(definterface ElemIdxToAddressFunction
  (^long idx_to_address [^long arg]))


;;This is the only one that will work with indirect addressing.
(defrecord GeneralElemIdxToAddr [rev-shape rev-strides rev-max-shape]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (ct-dims/elem-idx->addr rev-shape rev-strides rev-max-shape arg)))


(defrecord ElemIdxToAddr [^ints rev-shape ^ints rev-strides ^ints rev-max-shape]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (let [num-items (alength rev-shape)]
     (loop [idx (long 0)
            arg (long arg)
            offset (long 0)]
       (if (and (> arg 0)
                (< idx num-items))
         (let [next-max (aget rev-max-shape idx)
               next-stride (aget rev-strides idx)
               next-dim (aget rev-shape idx)
               max-idx (rem arg next-max)
               shape-idx (rem arg next-dim)]
           (recur (inc idx)
                  (quot arg next-max)
                  (+ offset (* next-stride shape-idx))))
         offset)))))


(defrecord SimpleElemIdxToAddr []
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   arg))


(defrecord SimpleBcastAddr [^long elem-count ^long bcast-amt]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (rem (quot arg bcast-amt)
        elem-count)))


(defrecord SimpleRepeatAddr [^long elem-count]
  ElemIdxToAddressFunction
  (^long idx_to_address [this ^long arg]
   (rem arg elem-count)))


(defn get-elem-dims->address
  ^ElemIdxToAddressFunction [dims max-shape]
  ;;Special cases here for speed
  (let [dense? (ct-dims/dense? dims)
        increasing? (ct-dims/access-increasing? dims)
        ;;Any indirect addressing?
        direct? (ct-dims/direct? dims)
        min-shape (drop-while #(= 1 %) (ct-dims/shape dims))]
    (cond
      ;;Special case for indexes that increase monotonically
      (and direct?
           (= (:shape dims)
              max-shape)
           dense?
           increasing?)
      (->SimpleElemIdxToAddr)
      ;;Special case for broadcasting a vector across an image (like applying bias).
      (and direct?
           (= (ct-dims/ecount dims)
              (apply max (ct-dims/shape dims)))
           dense?
           increasing?)
      (let [ec (ct-dims/ecount dims)
            ec-idx (long
                    (->> (map-indexed vector (ct-dims/left-pad-ones
                                              (ct-dims/shape dims) max-shape))
                         (filter #(= ec (second %)))
                         (ffirst)))
            broadcast-amt (long (apply * 1 (drop (+ 1 ec-idx) max-shape)))]
        (->SimpleBcastAddr ec broadcast-amt))
      (and direct?
           dense?
           increasing?
           (= min-shape
              (take-last (count min-shape) max-shape)))
      (->SimpleRepeatAddr (ct-dims/ecount dims))
      :else
      (let [{:keys [reverse-shape reverse-strides]}
            (ct-dims/->reverse-data dims max-shape)]
        (if direct?
          (->ElemIdxToAddr (int-array reverse-shape) (int-array reverse-strides)
                           (int-array (vec (reverse max-shape))))
          (do
            (->GeneralElemIdxToAddr (mapv (fn [item]
                                            (cond-> item
                                              (ct/tensor? item)
                                              ct/tensor->buffer))
                                          reverse-shape)
                                    reverse-strides
                                    (ct-utils/reversev max-shape))))))))


(defn max-shape-from-dimensions
  [& args]
  (-> (apply ct-dims/dimension-seq->max-shape args)
      :max-shape))
