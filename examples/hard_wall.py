import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from jester.fmt import compile_fmt, FourierGrid
from jester.sexpr import tokenize, Parser
from jester.functional import decode_functional
from jester.jax_eval import eval_sexp

# enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# load the functional from file
src = Path(__file__).with_name("rosenfeld.dft").read_text()


p = Parser(src)
sexp = p.parse()
fun = decode_functional(sexp)

grid = FourierGrid.create(
    shape=(128,128,128),
    lengths=(5.0, 5.0, 5.0)
)
compiled = compile_fmt(fun, grid)


R, rho_bulk, alpha = 0.5, 0.6, 0.02
Nx, Ny, Nz = grid.shape
dz = grid.spacing[2]
z = (jnp.arange(Nz) + 0.5) * dz

mask = ((z >= R) & (z <= grid.lengths[2] - R))[None, None, None, :]
rho0 = jnp.full((1, Nx, Ny, Nz), rho_bulk)
c1_bulk = compiled.direct_correlation(rho0)[0, 0, 0, 0]
beta_mu = jnp.log(rho_bulk) - c1_bulk

rho = rho0 * mask
for iteration in range(300):
    c1 = compiled.direct_correlation(rho)
    target = mask * jnp.exp(jnp.clip(beta_mu + c1, -50.0, 50.0))
    rho_new = (1.0 - alpha) * rho + alpha * target
    error = float(jnp.max(jnp.abs(rho_new - rho)))
    if iteration % 100 == 0:
        print(f"{iteration}: {error}")
    rho = rho_new
    if error < 1e-10:
        break

profile = np.asarray(rho[0].mean(axis=(0, 1)))
print(f"{iteration=}, {error=:.3e}, beta_mu={float(beta_mu):.8f}")


plt.plot(np.asarray(z), profile)
plt.axvline(R, ls="--")
plt.axvline(grid.lengths[2] - R, ls="--")
plt.xlabel("z")
plt.ylabel(r"$\rho(z)$")
plt.savefig("wall_profile.pdf")

