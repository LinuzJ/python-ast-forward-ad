Some playing around with pyhton ast functionality. Derivating a function `f` to `f_forward_ad` using automatic forward derivation through ast manipulations.

```py
def f_forward_ad(x: DualNum) -> DualNum
```

Then comparing the results with PyTorch and JAX alternatives for derivation.

Useful resources:

- Official documentation for Python AST: https://docs.python.org/3/library/ast.html
- Additional documentation for Python AST: https://greentreesnakes.readthedocs.io/en/latest/index.html
- Wiki article about Forward AD and dual numbers: https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers
