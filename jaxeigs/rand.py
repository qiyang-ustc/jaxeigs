import jax
import jax.numpy as jnp
import numpy as np
import warnings
from typing import Optional, Tuple, Callable, List, Text, Type, Any

def randn(shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None,
            seed: Optional[int] = None) -> jax.Array:
    if not seed:
        seed = np.random.randint(0, 2**63)
    key = jax.random.PRNGKey(seed)

    dtype = dtype if dtype is not None else np.dtype(np.float64)

    def cmplx_randn(complex_dtype, real_dtype):
        real_dtype = np.dtype(real_dtype)
        complex_dtype = np.dtype(complex_dtype)

        key_2 = jax.random.PRNGKey(seed + 1)
  
        real_part = jax.random.normal(key, shape, dtype=real_dtype)
        complex_part = jax.random.normal(key_2, shape, dtype=real_dtype)
        unit = (
            np.complex64(1j)
            if complex_dtype == np.dtype(np.complex64) else np.complex128(1j))
        return real_part + unit * complex_part

    if np.dtype(dtype) is np.dtype(jnp.complex128):
        return cmplx_randn(dtype, jnp.float64)
    if np.dtype(dtype) is np.dtype(jnp.complex64):
        return cmplx_randn(dtype, jnp.float32)

    return jax.random.normal(key, shape).astype(dtype)

def random_uniform(shape: Tuple[int, ...],
                    boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                    dtype: Optional[np.dtype] = None,
                    seed: Optional[int] = None) -> jax.Array:
    if not seed:
        seed = np.random.randint(0, 2**63)
    key = jax.random.PRNGKey(seed)

    dtype = dtype if dtype is not None else np.dtype(np.float64)

    def cmplx_random_uniform(complex_dtype, real_dtype):
        real_dtype = np.dtype(real_dtype)
        complex_dtype = np.dtype(complex_dtype)

        key_2 = jax.random.PRNGKey(seed + 1)

        real_part = jax.random.uniform(
            key,
            shape,
            dtype=real_dtype,
            minval=boundaries[0],
            maxval=boundaries[1])
        complex_part = jax.random.uniform(
            key_2,
            shape,
            dtype=real_dtype,
            minval=boundaries[0],
            maxval=boundaries[1])
        unit = (
            np.complex64(1j)
            if complex_dtype == np.dtype(np.complex64) else np.complex128(1j))
        return real_part + unit * complex_part

    if np.dtype(dtype) is np.dtype(jnp.complex128):
        return cmplx_random_uniform(dtype, jnp.float64)
    if np.dtype(dtype) is np.dtype(jnp.complex64):
        return cmplx_random_uniform(dtype, jnp.float32)

    return jax.random.uniform(
        key, shape, minval=boundaries[0], maxval=boundaries[1]).astype(dtype)
    