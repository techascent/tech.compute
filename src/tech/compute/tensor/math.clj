(ns tech.compute.tensor.math
  "Protocol to abstract implementations from the tensor library.  Tensors do not appear in at
  this level; at this level we have buffers, streams, and index systems.  This is intended to
  allow operations that fall well outside of the tensor definition to happen with clever use of
  the buffer and index strategy mechanisms.  In essence, the point is to make the kernels as
  flexible as possible so to allow extremely unexpected operations to happen without requiring
  new kernel creation.  In addition the tensor api should be able to stand on some subset of
  the possible combinations of operations available.")


(defprotocol TensorMath
  "Operations defined in general terms to enable the tensor math abstraction and to allow
  unexpected use cases outside of the tensor definition."
  (assign-constant! [stream tensor value]
    "Assign a constant value to a buffer. using an index strategy.")
  (assign! [stream dest src]
    "Assign to dest values from src using the appropriate index strategy.  Note that assignment
*alone* should be marshalling if both src and dst are on the same device.  So for the three
types used in the library: [:float :double :int] all combinations of assignment independent of
indexing strategy should be provided.
This function will not be called if dest and src are on different devices, memcpy semantics are
enforced for that case.")
  (unary-accum! [strea dest alph op]
    "dest[idx] = op(alpha * dest[idx]")
  (unary-op! [stream dest x alpha op]
    "dest[idx] = op( x[idx] * alpha )")
  (binary-accum-constant! [stream dest dst-alpha scalar operation reverse-operands?]
    "Binary operation where dest is involved in the computation.
dest[idx] = alpha * dest[idx] op scalar")

  (binary-op-constant! [stream dest x x-alpha scalar operation reverse-operands?]
    "Binary operation where dest is not involved in the computation.
dest[idx] = alpha * x[idx] op scalar")

  (binary-accum! [stream dest dest-alpha y y-alpha operation reverse-operands? dest-reduction?]
    "Binary operation where dest is involved in the computation.
dest[idx] = alpha * dest[idx] op y[idx]
reverse-operands?  Whether to reverse the operands.
dest-reduction? If the tensor library detects that dest is only written to once ever
then no CAS operation is required.  Else a CAS operation is potentially required as the destination
may be written to multiple times during the operation.")

  (binary-op! [stream dest x x-alpha y y-alpha operation]
    "Binary operation where dest is not involved in the computation.
dest[idx] = alpha * x[idx] op y[idx]")

  (ternary-op! [stream dest
                x x-alpha
                y y-alpha
                z z-alpha
                operation]
    "Apply ternary elementwise operation to args")

  (ternary-op-constant! [stream dest a a-alpha b b-alpha
                         constant operation arg-order]
    "Apply ternary elementwise operation to args and constant.
Argument order is specified by arg-order.")

  (ternary-op-constant-constant! [stream dest a a-alpha const-1 const-2 operation arg-order]
    "Apply ternary elementwise operation to args + 2 constants.
Argument order is specified by arg-order")

  (unary-reduce! [stream output input-alpha input op]
    "Reduction on 1 operand.")

  (gemm! [stream
          c-buf c-colstride
          trans-a? trans-b? alpha
          a-buf a-row-count a-col-count a-colstride
          b-buf b-col-count b-colstride
          beta]
    "Generalized matrix multiply.  In this case we don't pass in the index system
because gemm is not implemented in any system with anything like indirect addressing or
any other of the index system features aside from column stride.
c = alpha * (trans-a? a) * (trans-b? b) + beta * c")

  (rand! [stream dest distribution]
    "Generate a pool of random numbers over this distribution.  All elements of dest are assigned
to randomly out of the pool."))


(defprotocol LAPACK
  "Operations corresponding to the lapack set of functions."
  (cholesky-factorize! [stream dest-A upload]
    "dpotrf bindings.  Dest is both input and result argument.
dest: io argument, corresponding to matrix that is being factorized.
upload: :upper or :lower, store U or L.")
  (cholesky-solve! [stream dest-B upload A]
    "dpotrs bindings.
dest-B: Matrix to solve.
upload: :upper or :lower depending on if A is upper or lower.
A: cholesky-decomposed matrix A.")

  (LU-factorize! [stream dest-A dest-ipiv row-major?]
    "LU factorize with dest-a being the matrix to solve that receives the answer and
dest-ipiv being and integer tensor that receives the pivot indices.
column-major? indicates to factorize as if the matrix was provided in column-major form.
This is not the default and may impose a performance penalty if the underlying
system is column major and does not allow for changes.")

  (LU-solve! [stream dest-B trans A ipiv row-major?]
    "Solve using LU-factored A (possibly transposed) and pivot ary.  B is matrix to solve and will
contain the solution matrix.
Row-major implies a performance penalty for base lapack systems.
trans: one of - [:no-transpose :transpose :conjugate-transpose]")

  (singular-value-decomposition! [stream jobu jobvt A s U VT]
    "Computes the singular value decomposition (SVD) of a real
 M-by-N matrix A, optionally computing the left and/or right singular
 vectors. The SVD is written

      A = U * SIGMA * transpose(V)

 where SIGMA is an M-by-N matrix which is zero except for its
 min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
 are the singular values of A; they are real and non-negative, and
 are returned in descending order.  The first min(m,n) columns of
 U and V are the left and right singular vectors of A.

 Note that the routine returns V**T, not V."))
