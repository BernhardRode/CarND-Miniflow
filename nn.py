"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *


x, y, z = Input(), Input(), Input()
f = Add(x, y, z)
feed_dict_sum = {x: 4, y: 5, z: 10}
graph_sum = topological_sort(feed_dict_sum)
output_sum = forward_pass_old(f, graph_sum)
print("{} + {} + {} = {} (according to miniflow)".format(
    feed_dict_sum[x], feed_dict_sum[y], feed_dict_sum[z], output_sum))


a, b, c = Input(), Input(), Input()
m = Mul(a, b, c)
feed_dict_mul = {a: 4, b: 5, c: 10}
graph_mul = topological_sort(feed_dict_mul)
output_mul = forward_pass_old(m, graph_mul)
print("{} * {} * {} = {} (according to miniflow)".format(
    feed_dict_mul[a], feed_dict_mul[b], feed_dict_mul[c], output_mul))

inputs, weights, bias = Input(), Input(), Input()
f = Linear(inputs, weights, bias)
feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}
graph = topological_sort(feed_dict)
output = forward_pass_old(f, graph)
"""
Output should be:
12.7
"""
print(output)

X, W, b = Input(), Input(), Input()
f = Linear(X, W, b)
X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])
feed_dict = {X: X_, W: W_, b: b_}
graph = topological_sort(feed_dict)
output = forward_pass_old(f, graph)
"""
Output should be:
[[-9., 4.],
[-9., 4.]]
"""
print(output)
del X, W, b, f, X_, W_, b_


X, W, b = Input(), Input(), Input()
f = Linear(X, W, b)
g = Sigmoid(f)
X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])
feed_dict = {X: X_, W: W_, b: b_}
graph = topological_sort(feed_dict)
output = forward_pass_old(g, graph)
"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
print(output)

y, a = Input(), Input()
cost = MSE(y, a)
y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])
feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)
# forward pass
forward_pass(graph)
"""
Output should be:
23.4166666667
"""
print(cost.value)
