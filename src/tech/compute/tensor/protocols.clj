(ns tech.compute.tensor.protocols
  "General protocols specific for tensors.  Necessary to break the dependency chain")


(defprotocol PIsTensor
  (tensor? [item])
  (dense? [items]
    "Dense means: that the item is simply indexed (no fancy indexing) and that
the strides are uniformly decreasing *and* they densly pack the backing store.")
  (tensor->dimensions [item])
  (tensor->buffer [item]))


(defmacro not-a-tensor
  [item]
  `(throw (ex-info "not a tensor"
                   {:item-type (type ~item)})))


(extend-type Object
  PIsTensor
  (tensor? [item] false)
  (dense? [item] (not-a-tensor item))
  (tensor->dimensions [item] (not-a-tensor item))
  (tensor->buffer [item] (not-a-tensor item)))
