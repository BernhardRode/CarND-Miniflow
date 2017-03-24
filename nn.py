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
output_sum = forward_pass(f, graph_sum)
print("{} + {} + {} = {} (according to miniflow)".format(
    feed_dict_sum[x], feed_dict_sum[y], feed_dict_sum[z], output_sum))


a, b, c = Input(), Input(), Input()
m = Mul(a, b, c)
feed_dict_mul = {a: 4, b: 5, c: 10}
graph_mul = topological_sort(feed_dict_mul)
output_mul = forward_pass(m, graph_mul)
print("{} * {} * {} = {} (according to miniflow)".format(
    feed_dict_mul[a], feed_dict_mul[b], feed_dict_mul[c], output_mul))
