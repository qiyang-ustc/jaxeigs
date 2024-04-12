import jax
import jax.numpy as jnp
import numpy as np
import warnings
from typing import Optional, Tuple, Callable, List, Text, Type, Any
from eigensolver import jitted_functions

Tensor = Any
_CACHED_MATVECS = {}
_CACHED_FUNCTIONS = {}

def randn(shape: Tuple[int, ...],
            dtype: Optional[np.dtype] = None,
            seed: Optional[int] = None) -> Tensor:
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
                    seed: Optional[int] = None) -> Tensor:
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
    
"""
  Implicitly restarted Arnoldi method for finding the lowest
  eigenvector-eigenvalue pairs of a linear operator `A`.
  `A` is a function implementing the matrix-vector
  product.

  WARNING: This routine uses jax.jit to reduce runtimes. jitting is triggered
  at the first invocation of `eigs`, and on any subsequent calls
  if the python `id` of `A` changes, even if the formal definition of `A`
  stays the same.
  Example: the following will jit once at the beginning, and then never again:

  ```python
  import jax
  import numpy as np
  def A(H,x):
    return jax.np.dot(H,x)
  for n in range(100):
    H = jax.np.array(np.random.rand(10,10))
    x = jax.np.array(np.random.rand(10,10))
    res = eigs(A, [H],x) #jitting is triggerd only at `n=0`
  ```

  The following code triggers jitting at every iteration, which
  results in considerably reduced performance

  ```python
  import jax
  import numpy as np
  for n in range(100):
    def A(H,x):
      return jax.np.dot(H,x)
    H = jax.np.array(np.random.rand(10,10))
    x = jax.np.array(np.random.rand(10,10))
    res = eigs(A, [H],x) #jitting is triggerd at every step `n`
  ```

  Args:
    A: A (sparse) implementation of a linear operator.
        Call signature of `A` is `res = A(vector, *args)`, where `vector`
        can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
    args: A list of arguments to `A`.  `A` will be called as
      `res = A(initial_state, *args)`.
    initial_state: An initial vector for the algorithm. If `None`,
      a random initial `Tensor` is created using the `backend.randn` method
    shape: The shape of the input-dimension of `A`.
    dtype: The dtype of the input `A`. If no `initial_state` is provided,
      a random initial state with shape `shape` and dtype `dtype` is created.
    num_krylov_vecs: The number of iterations (number of krylov vectors).
    numeig: The number of eigenvector-eigenvalue pairs to be computed.
    tol: The desired precision of the eigenvalues. For the jax backend
      this has currently no effect, and precision of eigenvalues is not
      guaranteed. This feature may be added at a later point. To increase
      precision the caller can either increase `maxiter` or `num_krylov_vecs`.
    which: Flag for targetting different types of eigenvalues. Currently
      supported are `which = 'LR'` (larges real part) and `which = 'LM'`
      (larges magnitude).
    maxiter: Maximum number of restarts. For `maxiter=0` the routine becomes
      equivalent to a simple Arnoldi method.
  Returns:
    (eigvals, eigvecs)
      eigvals: A list of `numeig` eigenvalues
      eigvecs: A list of `numeig` eigenvectors
"""
def eigs(A: Callable,
          args: Optional[List] = None,
          initial_state: Optional[Tensor] = None,
          shape: Optional[Tuple[int, ...]] = None,
          dtype: Optional[Type[np.number]] = None,
          num_krylov_vecs: int = 50,
          numeig: int = 6,
          tol: float = 1E-8,
          which: Text = 'LR',
          maxiter: int = 20) -> Tuple[Tensor, List]:
    if args is None:
        args = []
    if which not in ('LR', 'LM'):
        raise ValueError(f'which = {which} is currently not supported.')

    if numeig > num_krylov_vecs:
        raise ValueError('`num_krylov_vecs` >= `numeig` required!')

    if initial_state is None:
        if (shape is None) or (dtype is None):
            raise ValueError("if no `initial_state` is passed, then `shape` and"
                        "`dtype` have to be provided")
        initial_state = randn(shape, dtype)

    if not isinstance(initial_state, (jnp.ndarray, np.ndarray)):
        raise TypeError("Expected a `jax.array`. Got {}".format(
            type(initial_state)))

    if A not in _CACHED_MATVECS:
        _CACHED_MATVECS[A] = jax.tree_util.Partial(jax.jit(A))

    if "imp_arnoldi" not in _CACHED_FUNCTIONS:
        imp_arnoldi = jitted_functions._implicitly_restarted_arnoldi(jax)
        _CACHED_FUNCTIONS["imp_arnoldi"] = imp_arnoldi

    eta, U, numits = _CACHED_FUNCTIONS["imp_arnoldi"](_CACHED_MATVECS[A], args,
                                                    initial_state,
                                                    num_krylov_vecs, numeig,
                                                    which, tol, maxiter,
                                                    jax.lax.Precision.DEFAULT)
    if numeig > numits:
        warnings.warn(
            f"Arnoldi terminated early after numits = {numits}"
            f" < numeig = {numeig} steps. For this value of `numeig `"
            f"the routine will return spurious eigenvalues of value 0.0."
            f"Use a smaller value of numeig, or a smaller value for `tol`")
    return eta, U