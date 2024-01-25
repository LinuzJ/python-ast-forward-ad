import ast
import random
import torch
import jax
import jax.numpy as np
from math import exp, cos, sin
from pprint import pprint
from collections import namedtuple
from numbers import Number


f_str = """
def f(x):
    return exp(x)**3 + cos(x) * x + 10**2
"""
print("Original function", f_str)
exec(f_str)

# Now `f` is defined
tree = ast.parse(f_str)


DualNum = namedtuple("DualNum", ["value", "derivative"])


def custom_exp(inp: DualNum):
    return DualNum(exp(inp.value), exp(inp.value) * inp.derivative)


def custom_cos(inp: DualNum):
    return DualNum(cos(inp.value), -sin(inp.value) * inp.derivative)


def custom_add(inp1: DualNum, inp2: DualNum):
    return DualNum(inp1.value + inp2.value, inp1.derivative + inp2.derivative)


def custom_mul(inp1: DualNum, inp2: DualNum):
    return DualNum(
        inp1.value * inp2.value,
        inp1.derivative * inp2.value + inp2.derivative * inp1.value,
    )


def custom_pow(inp: DualNum, k: Number):
    return DualNum(inp.value**k, inp.derivative * k * inp.value ** (k - 1))


# Main code for performing the forward automatic derivation
class ForwardDerivateTransformer(ast.NodeTransformer):
    def visit_Call(self, node) -> ast.Call:
        if node.func.id in ["cos", "exp"]:
            return ast.Call(
                func=ast.Name(id=f"custom_{node.func.id}", ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[],
            )

        raise Exception("Unknown Call")

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        if type(node.op) == ast.Add:
            return ast.Call(
                func=ast.Name(id="custom_add", ctx=ast.Load()),
                args=[self.visit(node.left), self.visit(node.right)],
                keywords=[],
            )
        if type(node.op) == ast.Mult:
            return ast.Call(
                func=ast.Name(id="custom_mul", ctx=ast.Load()),
                args=[self.visit(node.left), self.visit(node.right)],
                keywords=[],
            )
        elif type(node.op) == ast.Pow:
            return ast.Call(
                func=ast.Name(id="custom_pow", ctx=ast.Load()),
                args=[self.visit(node.left), node.right],
                keywords=[],
            )

        raise Exception("Unknown BinOp")

    def visit_Constant(self, node: ast.Constant) -> ast.Call:
        if isinstance(node.value, (float, int)):
            return ast.Call(
                func=ast.Name(id="DualNum", ctx=ast.Load()),
                args=[
                    node,
                    ast.Constant(value=0),
                ],
                keywords=[],
            )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = "f_forward_ad"
        if len(node.args.args) != 1:
            raise Exception(f"""Only one argument is supported, got {
                            len(node.args.args)}""")
        node.args.args[0].annotation = ast.Name(id="DualNum", ctx=ast.Load())

        return self.generic_visit(node)


# Create the ast tree for the f_forward_ad function
tree_ = ForwardDerivateTransformer().visit(tree)

f_forward_str = ast.unparse(tree_)

print("Updated function: \n", f_forward_str)
exec(f_forward_str)


# ============== TEST AGAINST PYTORCH AND JAX ==============
# Compute derivative using PyTorch
def t(xx):
    x = torch.tensor(xx, requires_grad=True)
    y = torch.exp(x) ** 3 + torch.cos(x) * x + 10**2
    y.backward()
    return y.item(), x.grad.item()


# Compute derivative using Jax
def j(xx):
    def f_for_jax(xx):
        return np.exp(xx) ** 3 + np.cos(xx) * xx + 10**2

    x_jax = np.array(xx)
    y_jax = f_for_jax(x_jax)

    grad_fn = jax.grad(f_for_jax)
    grad_jax = grad_fn(x_jax)
    return y_jax, grad_jax


print("\n")

seed_value = 42
random.seed(seed_value)
testing_values = [random.uniform(-3.0, 3.0) for _ in range(3)]

for x in testing_values:
    y_forward_ad, derivative_forward_ad = f_forward_ad(DualNum(x, 1))
    y_torch, derivative_torch = t(x)
    y_jax, derivative_jax = j(x)

    # Use a tolerance-based comparison
    assert np.isclose(y_forward_ad, np.array(
        [y_torch, y_jax]), atol=1e-05).all(), \
        f"""y not equal for x={x}, y_forward_ad={
        y_forward_ad}, y_torch={y_torch}, y_jax={y_jax}"""

    # Make sure the derivative is the same for f_forward_ad, pytorch, and jax
    assert np.isclose(derivative_forward_ad, np.array(
        [derivative_torch, derivative_jax]), atol=1e-05).all(), \
        f"derivative not equal for x={x}, derivative_forward_ad={derivative_forward_ad}, " \
        f"derivative_torch={derivative_torch}, derivative_jax={derivative_jax}"

# If no tests failed -> success
print("Success!")
