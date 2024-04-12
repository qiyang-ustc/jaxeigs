import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["cpu_eig"]

def cpu_eig_host(H):
    res = np.linalg.eig(H)
    print(res)
    return res

def cpu_eig(H):
    result_shape = (jax.ShapeDtypeStruct(H.shape[0:1], H.dtype),
                    jax.ShapeDtypeStruct(H.shape, H.dtype))
    return jax.pure_callback(cpu_eig_host, result_shape, H)