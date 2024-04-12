import jax
import jax.numpy as jnp
from jaxeig import eigs
from jax import config
config.update("jax_enable_x64", True)

if __name__ == "__main__":
    m = 10
    A = jax.random.uniform(jax.random.PRNGKey(42),(m,m))
    b = jax.random.uniform(jax.random.PRNGKey(41),(m,))
    def mapA(x): return A@x
    res = eigs(mapA, initial_state = b, numeig=1, num_krylov_vecs = 5)
    print(res[0],res[1][0])
    
    A @ res[1][0] / res[1][0]