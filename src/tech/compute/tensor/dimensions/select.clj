(ns tech.compute.tensor.dimensions.select
  "Selecting subsets from a larger set of dimensions leads to its own algebra."
  (:require [tech.compute.tensor.protocols :refer [tensor? dense?]]
            [clojure.core.matrix :as m]
            [tech.datatype.core :as dtype]))

(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defn reversev
  [item-seq]
  (if (vector? item-seq)
    (let [len (count item-seq)
          retval (transient [])]
      (loop [idx 0]
        (if (< idx len)
          (do
            (conj! retval (item-seq (- len idx 1)))
            (recur (inc idx)))
          (persistent! retval))))
    (vec (reverse item-seq))))


(def monotonic-operators
  {:+ <
   :- >})


(defn- classify-sequence-type
  [item-seq]
  ;;Normalize this to account for single digit numbers
  (let [item-seq (if (number? item-seq)
                   [item-seq]
                   item-seq)]
    (when-not (seq item-seq)
      (throw (ex-info "Nil sequence in check monotonic" {})))
    (let [n-elems (count item-seq)
          first-item (long (first item-seq))
          last-item (long (last item-seq))
          min-item (min first-item last-item)
          max-item (max first-item last-item)
          retval {:min-item min-item
                  :max-item max-item}]
      (if (= n-elems 1)
        (assoc retval :type :+)
        (let [mon-op (->> monotonic-operators
                          (map (fn [[op-name op]]
                                 (when (apply op item-seq)
                                   op-name)))
                          (remove nil?)
                          first)]
          (if (and (= n-elems
                      (+ 1
                         (- max-item
                            min-item)))
                   mon-op)
            (assoc retval :type mon-op)
            (assoc retval :sequence (vec item-seq))))))))


(defn is-classified-sequence?
  [item]
  (map? item))


(defn- classified-sequence->sequence
  [{:keys [min-item max-item type sequence]}]
  (->> (if sequence
         sequence
         (case type
           :+ (range min-item (inc (long max-item)))
           :- (range max-item (dec (long min-item)) -1)))
       ;;Ensure dtype/get-value works on result.
       vec))


(defn- combine-classified-sequences
  "Room for optimization here.  But simplest way is easiest to get correct."
  [source-sequence select-sequence]
  (let [last-valid-index (if (:type source-sequence)
                           (- (long (:max-item source-sequence))
                              (long (:min-item source-sequence)))
                           (- (count (:sequence source-sequence))
                              1))]
    (when (> (long (:max-item select-sequence))
             last-valid-index)
      (throw (ex-info "Select argument out of range" {:dimension source-sequence
                                                      :select-arg select-sequence}))))
  (let [source-sequence (classified-sequence->sequence source-sequence)]
    (->> (classified-sequence->sequence select-sequence)
         (map #(get source-sequence (long %)))
         classify-sequence-type)))


(defn- expand-dimension
  [dim]
  (cond (number? dim)
        {:type :+
         :min-item 0
         :max-item (- (long dim) 1)}
        (map? dim) dim
        (sequential? dim) (classify-sequence-type dim)
        (tensor? dim) dim
        :else
        (throw (ex-info "Failed to recognize dimension type"
                        {:dimension dim}))))


(defn- verify-tensor-indexer
  [select-arg]
  (let [tens-shape (dtype/shape select-arg)]
    (when-not (= 1 (count tens-shape))
      (throw (ex-info "Only tensor of rank 1 can be used as indexes in other tensors"
                      {:select-arg-shape tens-shape})))
    (when-not (dense? select-arg)
      (throw (ex-info "Only dense tensors can be used as indexes in other tensors" {})))
    (when-not (= :int32 (dtype/get-datatype select-arg))
      (throw (ex-info "Index tensors must be int32 datatype"
                      {:tensor-datatype (dtype/get-datatype select-arg)})))
    select-arg))


(defn- expand-select-arg
  [select-arg]
  (cond
    (number? select-arg) (classify-sequence-type select-arg)
    (map? select-arg) select-arg
    (sequential? select-arg) (classify-sequence-type select-arg)
    (= :all select-arg) select-arg
    (= :lla select-arg) select-arg
    (tensor? select-arg) (verify-tensor-indexer select-arg)
    :else
    (throw (ex-info "Unrecognized select argument type"
                    {:select-arg select-arg}))))


(defn- reverse-classified-sequence
  [{:keys [type sequence] :as item}]
  (if sequence
    (assoc item :sequence
           (reversev sequence))
    (assoc item :type
           (if (= type :+)
             :-
             :+))))


(defn apply-select-arg-to-dimension
  "Given a dimension and select argument, create a new dimension with
the selection applied."
  [dim select-arg]
  ;;Dim is now a map or a tensor
  (let [dim (expand-dimension dim)
        ;;Select arg is now a map, a keyword, or a tensor
        select-arg (expand-select-arg select-arg)
        dim-type (cond (map? dim) :classified-sequence
                       (tensor? dim) :tensor)
        select-type (cond (map? select-arg) :classified-sequence
                          (tensor? select-arg) :tensor
                          (keyword? select-arg) :keyword)]
    (cond
      (= :tensor select-type)
      (do
        (when-not (and (= :classified-sequence dim-type)
                       (= :+ (:type dim))
                       (= 0 (long (:min-item dim))))
          (throw (ex-info "Can only use tensor indexers on monotonically incrementing dimensions"
                          {:dim dim
                           :select-arg select-arg}))
          select-arg))
      (= :all select-arg)
      dim
      (= :lla select-arg)
      (case dim-type
        :tensor (throw (ex-info "Can not reverse tensor indexers"
                                {:dimension dim :select-arg select-arg}))
        :classified-sequence (reverse-classified-sequence dim))
      (= :tensor dim-type)
      (throw (ex-info "Only :all select types are supported on tensor dimensions"
                      {:dimension dim
                       :select-arg select-arg}))
      :else
      (do
       ;;select arg and dim are classified sequences
       (assert (and (= dim-type :classified-sequence)
                    (= select-type :classified-sequence))
               "Internal logic failure")
       (combine-classified-sequences dim select-arg)))))


(defn dimensions->simpified-dimensions
  "Given the dimensions post selection, produce a new dimension sequence combined with
  an offset that lets us know how much we should offset the base storage type.
  Simplification is important because it allows a backend to hit more fast paths.
Returns:
{:dimension-seq dimension-seq
:offset offset}"
  [dimension-seq stride-seq]
  (let [[dimension-seq offset]
        (reduce (fn [[dimension-seq offset] [dimension stride]]
                  (cond
                    (map? dimension)
                    ;;Shift the sequence down and record the new offset.
                    (let [{:keys [type min-item max-item sequence]} dimension
                          max-item (- (long max-item) (long min-item))
                          new-offset (+ (long offset)
                                        (* (long stride)
                                           (long (:min-item dimension))))
                          min-item 0
                          dimension (cond-> (assoc dimension
                                                   :min-item 0
                                                   :max-item max-item)
                                      sequence
                                      (assoc :sequence (mapv (fn [idx]
                                                               (- (long idx) (long min-item)))
                                                             sequence)))
                          ;;Now simplify the dimension if possible
                          dimension (cond
                                      (= (long min-item) (long max-item))
                                      1
                                      (= :+ type)
                                      (+ 1 max-item)
                                      sequence
                                      (:sequence dimension)
                                      :else
                                      dimension)]
                      [(conj dimension-seq dimension) new-offset])
                    (tensor? dimension)
                    [(conj dimension-seq dimension) offset]))
                [[] 0]
                (map vector dimension-seq stride-seq))]
    {:dimension-seq dimension-seq
     :offset offset}))


(defn classified-sequence->elem-idx
  ^long [{:keys [type min-item max-item sequence] :as dim} ^long shape-idx]
  (let [min-item (long min-item)
        max-item (long max-item)
        last-idx (- max-item min-item)]
    (when (> shape-idx last-idx)
      (throw (ex-info "Element access out of range"
                      {:shape-idx shape-idx
                       :dimension dim})))
    (if (= :+ type)
      (+ min-item shape-idx)
      (- max-item shape-idx))))
