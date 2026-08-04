import jax.numpy as jnp
import jax

from jester.sexpr import tokenize, Parser
from jester.functional import decode_functional
from jester.jax_eval import eval_sexp
from jester.fmt import compile_fmt, FourierGrid


jax.config.update("jax_enable_x64", True)


with open("rosenfeld.dft") as file:
    src = file.read()


p = Parser(src)
sexp = p.parse()

fun = decode_functional(sexp)

grid = FourierGrid.create(
    shape=(64,64,64),
    lengths=(4.0, 4.0, 4.0)
)
compiled = compile_fmt(fun, grid)

rho = jnp.ones(grid.shape)

Fex = compiled.excess_energy(rho)
c1 = compiled.direct_correlation(rho)

print(Fex, c1.shape)