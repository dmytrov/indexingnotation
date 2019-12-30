# indexingnotation - computing with tensors without much pain

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


Example of a reusable covariance function:

    def covar(a, b, along):
        a = a - a.reduce(mean, along)
        b = b - b.reduce(mean, along)
        return (a * b).reduce(sum, along) / a.shape()[along]

    a = Named(np.reshape(np.random.uniform(size=5*3), [5, 3]), ("t", "a"))
    b = Named(np.reshape(np.random.uniform(size=5*4), [5, 4]), ("t", "b"))
    print(covar(a, b, along="t"))
    
