(ns tech.tensor.utils)


(defmulti dtype-cast
  "Runtime datatype casting from a scalar value to a scalar value"
  (fn [elem dtype]
    dtype))

(defmethod dtype-cast :float64
  [elem dtype]
  (double elem))

(defmethod dtype-cast :float32
  [elem dtype]
  (float elem))

(defmethod dtype-cast :int64
  [elem dtype]
  (long elem))

(defmethod dtype-cast :int32
  [elem dtype]
  (int elem))

(defmethod dtype-cast :int16
  [elem dtype]
  (short elem))

(defmethod dtype-cast :int8
  [elem dtype]
  (byte elem))
