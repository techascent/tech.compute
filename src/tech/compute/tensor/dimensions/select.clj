(ns tech.compute.tensor.dimensions.select
  "Selecting subsets from a larger set of dimensions leads to its own algebra."
  (:require [tech.compute.tensor.protocols :refer [tensor? dense?]]
            [tech.compute.tensor.dimensions.shape :as shape]
            [clojure.core.matrix :as m]
            [tech.datatype.core :as dtype]))

(set! *unchecked-math* :warn-on-boxed)
(set! *warn-on-reflection* true)


(defn- expand-dimension
  [dim]
  (cond (number? dim)
        {:type :+
         :min-item 0
         :max-item (- (long dim) 1)}
        (map? dim) dim
        (sequential? dim) (shape/classify-sequence dim)
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
    (number? select-arg) (shape/classify-sequence select-arg)
    (map? select-arg) select-arg
    (sequential? select-arg) (shape/classify-sequence select-arg)
    (= :all select-arg) select-arg
    (= :lla select-arg) select-arg
    (tensor? select-arg) (verify-tensor-indexer select-arg)
    :else
    (throw (ex-info "Unrecognized select argument type"
                    {:select-arg select-arg}))))


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
        :classified-sequence (shape/reverse-classified-sequence dim))
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
       (shape/combine-classified-sequences dim select-arg)))))


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
