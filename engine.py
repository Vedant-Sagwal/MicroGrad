import math
class Value:
    def __init__(self, data, _children = (), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward() :
            self.grad += 1 * out.grad #needs to plus equals(+=) because derivative has to accumulate if a variable is used multiple times
            other.grad += 1 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward() : 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other) :  # does other  * self
        return self * other
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self, ), f"**{other}")
        def _backward():
            self.grad += out.grad * (other * (self.data ** (other - 1)))
        out._backward = _backward
        return out
    def __truediv__(self, other):
        return self * (other ** -1) 
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
        
    def tanh(self):
        n = self.data
        t = (math.exp(2 *n) - 1) / (math.exp(2*n) + 1)
        out = Value(t, (self, ), "tanh")
        def _backward() : 
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), _op = "exp")
        def _backward():
            self.grad += math.exp(x) * out.grad
        out._backward = _backward
        return out
    def backward(self):
        #since calling _backward repeatedly makes no sense so use topo sort and call for _backward in that order.
        topo = []
        visited = set()
        def topoSort(node):
            if node not in visited:
                visited.add(node)
            for children in node._prev:
                if (children not in visited):
                    topoSort(children)
            topo.append(node)
        topoSort(self)
        self.grad = 1.0
        for i in reversed(topo):
            i._backward()
