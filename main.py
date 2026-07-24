from lisp_tokenizer import tokenize
from lisp_parser import Parser
from dsl import decode_functional
from eval_jax import eval_sexp
from fmt_jax import compile_fmt, FourierGrid

import jax.numpy as jnp

#src = "(+ (* n1 n2) (/ n2 3))"



with open("rosenfeld.dft") as file:
    src = file.read()


p = Parser(src)
sexp = p.parse()

fun = decode_functional(sexp)

grid = FourierGrid.create(
    shape=(64,64,64),
    lengths=(2.0, 2.0, 2.0)
)
compiled = compile_fmt(fun, grid)

rho = jnp.ones(grid.shape)

Fex = compiled.excess_energy(rho)
c1 = compiled.direct_correlation(rho)

print(Fex, c1.shape)


