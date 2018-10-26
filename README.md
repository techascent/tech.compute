# tech.compute


[![Clojars Project](https://clojars.org/techascent/tech.compute/latest-version.svg)](https://clojars.org/techascent/tech.compute)


Library designed to provide a generic compute abstraction to allow
some level of shared implementation between a cpu, cuda, openCL,
webworkers, etc. design.  This system is not specific to neural networks
nor it is specific to doing linear algebra but is rather the basis
to implement any generic algorithm across a range of environments but
targetting environments where there is a distinct transfer step between
the main computing system (called the host) and some external compute
system (called the device).  The primitives of this layer are carefully
chosen to be implementable across a wide range of different 'devices' such
that a unified codebase and run unmodified in the various potential runtimes
listed above and this *includes* operations like:

* Offsetting device buffers to create new buffers.  Thus allows efficient implementation
of pooling algorithms or buffer coalescing where the user desires to create 1 large
buffer and the create a series of smaller buffers piecemeal later.  Sub-buffers are not
distiguishable to the implementations from the original source buffers.
* Initializing buffers to fixed values.
* Transfer of data between the host and device.
* Overlapping transfer with device compute using multiple 'streams' of execution.
* Synchronizing stream execution with either the host system or with another on-device
  stream.


There is a generalized underlying compute layer that works across 6 primitive datatypes
in the JVM - bytes, shorts, ints, longs, floats and doubles - along with 4 datatypes
(unsigned versions of the integer types above). This compute layer contains the
protocols that define an interface to a compute driver, either cpu or cuda at this
time. This interface contains ways to move data from the host architecture to the
specific device architecture.


Components of the compute abstraction:

### [driver.clj](src/tech/compute/driver.clj)

(ns documentation)
```clojure
  "Base set of protocols required to move information from the host to the device as well as enable some form of computation on a given device.  There is a cpu implementation provided for reference.

  Base datatypes are defined:
   * Driver: Enables enumeration of devices and creation of host buffers.
   * Device: Creates streams and device buffers.
   * Stream: Stream of execution occuring on the device.
   * Buffer: Lightly typed (java primitive types at this time) heavy on-device buffer.
```


This layer defines operations you would expect to find on any buffer in a C-based
language such as:

Buffers allow memcpy, offsetting, upload to device and download from device.
Buffers can be offset to produce a new (shorter) buffer with a different base address.


There are two concepts used in this file that are not defined:

1. [resource management](https://github.com/tech-ascent/tech.resource)
2. [datatype](https://github.com/techascent/tech.datatype)

Resource management allows stack-based resource algorithms similar to the RAII concept
in c++.  This is important to ensure that using GPU resources is as pleasant and as
forgiving as possible.


The datatype library allows identification and efficient copying of data into packed
sequential buffers, including marshalling copies where we want to say copy a buffer of
bytes into a buffer of floats.  This library is very carefully written to allow
efficient upload/download characteristics so client code does not have unnecessary
bottlenecks around moving data to or from a compute device while still allowing users
complete flexibility with regards to their choice of data used in their systems external
to the compute system.


This combination of systems (driver, datatype, and resource) are designed to work
together to enable a generic computing abstraction, not specific to linear algebra or
neural networks.  Should clojure become a language where people are expressing a wide
range of algorithms on the GPU, CPU, and perhaps in the web then this would be an
appropriate substrate to enable efficient data management for those algorithms.


There is one default driver provided for a pure-cpu implementation

1. [cpu](src/tech/compute/cpu/driver.clj)

### Maths!


Included in this system is a simple, generic math [library](src/tech/compute/tensor.clj)
useful across a range of devices.  This is not meant to be a comprehensive solution but
rather enough of a solution as to drastically cut down on the amount of engineering
required to move a problem onto those other devices and then use potentially custom code
or other libraries once on those devices.  As such it is engineered with a focus on
simplicity of backend implementation and generality of operations.


[Here](docs/tensor.md) is a document describing it in more detail.


## License

Copyright © 2018 TechAscent, LLC

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
