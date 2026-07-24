import jax.numpy as jnp

def rosenfeld_bulk_phi(rho, radius):
    eta = (
        (4.0 / 3.0)
        * jnp.pi
        * radius**3
        * rho
    )

    return rho * (
        -jnp.log1p(-eta)
        + 3.0 * eta / (1.0 - eta)
        + 1.5 * eta**2 / (1.0 - eta) ** 2
    )


def test_uniform_bulk_free_energy(compiled, grid, radius):
    rho0 = 0.25
    rho = jnp.full((1, *grid.shape), rho0)

    actual_density = compiled.excess_energy_density(rho)
    expected_density = rosenfeld_bulk_phi(rho0, radius)

    assert jnp.allclose(
        actual_density,
        expected_density,
        rtol=1e-12,
        atol=1e-12,
    )

    expected_energy = expected_density * jnp.prod(
        jnp.asarray(grid.lengths)
    )

    assert jnp.allclose(
        compiled.excess_energy(rho),
        expected_energy,
        rtol=1e-12,
        atol=1e-12,
    )
