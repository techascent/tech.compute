(ns tech.compute.tensor.protocols
  "Protocols specific for tensors.  Necessary to break the dependency chain")


(defprotocol PIsTensor
  (tensor? [item])
  (dense? [items]
    "Dense means: that the item is simply indexed (no fancy indexing) and that
the strides are uniformly decreasing *and* they densly pack the backing store."))


(extend-type Object
  PIsTensor
  (tensor? [item] false)
  (dense? [item] (throw (ex-info "Unimplemented" {}))))
