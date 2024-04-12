import jax


if __name__ == "__main__":
    backend = JaxBackend()
    m = 100
    A = jax.random.normal(jax.random.PRNGKey(42),(m,m))
    b = jax.random.normal(jax.random.PRNGKey(41),(m,))
    def mapA(x): return A@x
    backend.eigs(mapA,initial_state = b)