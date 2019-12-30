""" Basic named axis implementation.
    
    Requires python 3

    Author: Dmytro Velychko, 
            Philipps University of Marburg
            velychko@staff.uni-marburg.de
"""

import textwrap
from collections import namedtuple
import numpy as np


class Map(object):
    pass

class NegOp(Map):
    @staticmethod
    def __call__(x):
        return -x
      
class Selector(Map):
    pass

class SliceSelector(Map):
    pass

class ConditionSelector(Map):
    pass

class ReductionOp(Map):
    pass

class SumReductionOp(Map):
    @staticmethod
    def __call__(x, axes):
        return np.sum(x, axis=axes, keepdims=False)

class ProdReductionOp(Map):
    @staticmethod
    def __call__(x, axes):
        return np.prod(x, axis=axes, keepdims=False)

class MeanReductionOp(Map):
    @staticmethod
    def __call__(x, axes):
        return np.mean(x, axis=axes, keepdims=False)

class CrossOperation(Map):
    pass

class ProdCrossOperation(Map):
    @staticmethod
    def __call__(x, y):
        return x * y

neg = NegOp()
sum = SumReductionOp()
prod = ProdReductionOp()
mean = MeanReductionOp()

el_prod = ProdCrossOperation()


class Named(object):
    def __init__(self, tensor, names):
        assert len(tensor.shape) == len(names)
        assert isinstance(names, tuple)
        self._tensor = tensor  # actual data
        self._names = names  # axes names


    def shape(self):
        return {n: s for n, s in zip(self._names, self._tensor.shape)}


    def map(self, op):
        """ Apply a unitary mapping operation for every tensor element
        """
        return Named(op(self._tensor), self._names)


    def reduce(self, op, names):
        """ Remove axis by applying a commutative operation along it
        """
        axes = tuple((i for i, name in enumerate(self._names) if name in names))
        new_tensor = op(self._tensor, axes)
        new_names = tuple((name for name in self._names if name not in names))
        return Named(new_tensor, new_names)


    def elementwise(self, op, other):
        """ Apply elementwise operation between self and other.
            Resulting tensor axes names is join set of the argument tensor's axes
        """
        if not isinstance(other, Named):
            other = Named(np.array(other), ())
        new_names = self.join_names(self._names, other._names)
        new_tensor = op(self.expand(new_names)._tensor, other.expand(new_names)._tensor)
        return Named(new_tensor, new_names)


    def cross(self, other, name):
        """ Cartesian product of tensors by new axis name.
            Resulting tensor axes names is join set of the argument tensor's axes
            with added new axis name.
            Shared names dimensions shoud match
        """
        new_names = self.join_names(self._names, other._names)
        expanded_self = self.expand(new_names)._tensor
        expanded_other = other.expand(new_names)._tensor

        new_shape = np.max([expanded_self.shape, expanded_other.shape], axis=0)
        expanded_self = np.broadcast_to(expanded_self, new_shape)
        expanded_other = np.broadcast_to(expanded_other, new_shape)
        new_tensor = np.stack([expanded_self, expanded_other], axis=0)
        return Named(new_tensor, (name,) + new_names)


    def split(self, name):
        """ Splits and removes the axis
            Return a list of named tensors
        """
        new_names = tuple((n for n in self._names if n != name))
        for sliced in self.expand((name,) + new_names)._tensor:
            yield Named(sliced, new_names)


    @staticmethod
    def merge(tensors, name):
        """ Creates a new axis and merges tensors along it
        """
        assert len(tensors) == 2
        t1, t2 = tensors
        assert Named.contains_names(t1._names, t2._names)

        new_names = (name,) + t1._names
        new_tensor = np.concatenate([t1.expand(new_names)._tensor, t2.expand(new_names)._tensor])
        return Named(new_tensor, new_names)


    def expand(self, names):
        """ Expand by new axes in the exact order provided.
            If no new names - then just reorders the axes.
            The resulting tensor axes names are exactly $names$
        """
        assert Named.contains_names(names, self._names)
        additional_names = Named.diff_names(names, self._names)
        
        new_names = additional_names + self._names  # insert in front
        new_tensor = self._tensor
        for i in range(len(additional_names)):
            new_tensor = np.expand_dims(new_tensor, axis=0)  # insert in front
        
        ind = [new_names.index(n) for n in names]
        new_tensor = np.transpose(new_tensor, ind)
        return Named(new_tensor, names)


    def select(self, selectorop, names):
        inds = tuple((self._names.index(name) for name in names))
        
        new_tensor = selectorop(self._tensor, inds)
        new_names = "-".join(names)
        return Named(new_tensor, new_names)
        

    def __getitem__(self, slices):
        """ Slice the tensor according to the slices dictionary
        """
        assert isinstance(slices, dict) 
        slices = tuple((slices[key] if key in slices else slice(None) for key in self._names))
        new_names = tuple((name for name, s in zip(self._names, slices) 
                if isinstance(s, (slice, list, tuple, np.ndarray))))
        return Named(self._tensor[slices], new_names)


    def append(self, other, name):
        """ Append to existing axis
        """
        ind = self._names.index(name)
        return Named(np.concatenate([self._tensor, other._tensor], ind), self._names)


    def rename(self, pairs):
        """ Renames axes {old: new} for every pair
        """
        names = list(self._names)
        for key, val in pairs.items():
            i = names.index(key)
            names[i] = val
        new_names = tuple(names)
        return Named(self._tensor, new_names)


    @staticmethod
    def join_names(a, b):
        """ Set join
        """
        return tuple(set(a).union(set(b)))

    
    @staticmethod
    def intersect_names(a, b):
        """ Set join
        """
        return tuple(set(a).intersection(set(b)))


    @staticmethod
    def diff_names(a, b):
        """ Set difference a-b
        """
        return tuple(set(a).difference(set(b)))


    @staticmethod
    def contains_names(a, b):
        """ Check if a contains b
        """
        return set(a).issuperset(set(b))

    @staticmethod
    def same_names(a, b):
        """ Check if a and b are the same
        """
        return set(a) == set(b)


    def __add__(self, other):
        return self.elementwise(lambda a, b: a+b , other)

    def __sub__(self, other):
        return self.elementwise(lambda a, b: a-b , other)

    def __mul__(self, other):
        return self.elementwise(lambda a, b: a*b , other)

    def __truediv__(self, other):
        return self.elementwise(lambda a, b: a/b , other)

    def __radd__(self, other):
        if not isinstance(other, Named):
            other = Named(np.array(other), ())
        return other + self

    def __rsub__(self, other):
        if not isinstance(other, Named):
            other = Named(np.array(other), ())
        return other - self

    def __rmul__(self, other):
        if not isinstance(other, Named):
            other = Named(np.array(other), ())
        return other * self

    def __rtruediv__(self, other):
        if not isinstance(other, Named):
            other = Named(np.array(other), ())
        return other / self    
    
    def __neg__(self):
        return self.map(neg)


    def __repr__(self):
        meta = ", ".join(["{}: {}".format(n, s) for n, s in zip(self._names, self._tensor.shape)])
        res = "Named ({})".format(meta) + ":\n" + textwrap.indent(str(self._tensor), "  ")
        return res
        


if __name__ == "__main__":
    print("--- Reduction ---")
    x = Named(np.ones([2, 3, 4]), ("a", "b", "c"))
    print("x =", x)
    y = x.reduce(sum, "c")
    print("x.sum('c') =", y)

    print("--- Expansion ---")
    z = y.expand(("<", "b", "-", "a", ">"))
    print(z)

    print("--- Elementwise operation ---")
    x = Named(np.ones([2, 3]), ("a", "b"))
    y = Named(np.ones([3, 4]), ("b", "c"))
    z = x.elementwise(el_prod, y)
    print(z)

    print("--- Outer product ---")
    x = Named(np.arange(3), ("x",))
    y = Named(np.arange(4), ("y",))
    z = x.elementwise(el_prod, y)
    print(z)

    print("--- Rename axis ---")
    x = Named(np.ones([2, 3]), ("a", "b"))
    y = x.rename({"a": "new_a"})
    print(y)

    print("--- Inner product ---")
    x = Named(np.arange(3), ("x",))
    print("x =", x)
    z = x.elementwise(el_prod, x).reduce(sum, "x")
    print("x dot x =", z)

    print("--- Self outer product ---")
    x = Named(np.arange(3), ("x",))
    print("x =", x)
    z = x.elementwise(el_prod, x.rename({"x": "x'"}))
    print("x*x' =", z)

    print("--- Cross ---")
    x = Named(np.arange(3), ("x",))
    z = x.cross(x.rename({"x": "x'"}), "choice")
    print(z)

    y = Named(np.arange(4), ("y",))
    z = x.cross(y, "choice")
    print(z)

    print("--- Splitting ---")
    for i, t in enumerate(z.split("choice")):
        print("Splited by choice =", i, t)

    print("--- Merging ---")
    z = Named.merge([t for t in z.split("choice")], name="merged")
    print(z)

    print("--- Operators ---")
    x = Named(np.arange(3), ("x",))
    y = Named(np.arange(2), ("y",))
    print(x * x)
    print(x + x * y)
    print(x * x.rename({"x": "x'"}))

    print("--- Generic function ---")
    def var(x, along):
        x_centered = x - x.reduce(mean, along)
        return (x_centered * x_centered).reduce(sum, along) / x.shape()[along]

    x = Named(np.reshape(np.random.uniform(size=3*4), [3, 4]), ("t", "x"))
    print(x)
    print(var(x, along="t"))

    def covar(a, b, along):
        a = a - a.reduce(mean, along)
        b = b - b.reduce(mean, along)
        return (a * b).reduce(sum, along) / a.shape()[along]

    a = Named(np.reshape(np.random.uniform(size=5*3), [5, 3]), ("t", "a"))
    b = Named(np.reshape(np.random.uniform(size=5*4), [5, 4]), ("t", "b"))
    print(covar(a, b, along="t"))

    print("--- Scalar operations ---")
    x = Named(np.arange(3), ("x",))
    print(2 + x + 3)
    print(2 - x)
    print(2 * x)
    print(2 / x)
    print(-x - 1)

    print("--- Slices ---")
    x = Named(np.reshape(np.random.uniform(size=5*3), [5, 3]), ("a", "b"))
    print(x[{"a": 2, "b": 1}])
    print(x[{"a": slice(3), "b": 1}])
    print(x[{"a": slice(3), "b": slice(2)}])
    print(x[{"b": slice(2)}])
    print(x[{"b": [1,0]}])
    
    print("--- Append ---")
    x = Named(np.reshape(np.random.uniform(size=5*3), [5, 3]), ("a", "b"))
    print(x[{"a": slice(3)}].append(x[{"a": slice(3)}], "b"))
    
    