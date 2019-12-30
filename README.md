# indexingnotation
Human readable reusable indexing notation for tensor operations.

The relics of the past - fortran-like numpy and its derivatives equipped with autodiff (pytorch, tensorflow, etc) - will be inevitably replaced by a better and more expressive notation for tensor operations. 
 
Named axes finally bring reusability and readability into the code.

Supports:
 - axis reduction
 - axis renaming
 - automatic expansion (matching)
 - elementwise operations
 - inner product, outer product, Kronecker product
 - tensor split/merge
 - basic operators
 - numpy-like named slices
 - generic functions
 
Todo:
 - flexible filter to split tensors
 - more elementwise operations
 - pytorch wrapper
