import jax
import functools
from typing import Optional, Tuple, Callable, List, Text, Type, Any, Sequence
import types
from .gram_schmit import _iterative_classical_gram_schmidt

def _generate_jitted_eigsh_lanczos(jax: types.ModuleType) -> Callable:
    """
    Helper function to generate jitted lanczos function used
    in JaxBackend.eigsh_lanczos. The function `jax_lanczos`
    returned by this higher-order function has the following
    call signature:
    ```
    eigenvalues, eigenvectors = jax_lanczos(matvec:Callable,
                                                                        arguments: List[Tensor],
                                                                        init: Tensor,
                                                                        ncv: int,
                                                                        neig: int,
                                                                        landelta: float,
                                                                        reortho: bool)
    ```
    `matvec`: A callable implementing the matrix-vector product of a
    linear operator. `arguments`: Arguments to `matvec` additional to
    an input vector. `matvec` will be called as `matvec(init, *args)`.
    `init`: An initial input vector to `matvec`.
    `ncv`: Number of krylov iterations (i.e. dimension of the Krylov space).
    `neig`: Number of eigenvalue-eigenvector pairs to be computed.
    `landelta`: Convergence parameter: if the norm of the current Lanczos vector

    `reortho`: If `True`, reorthogonalize all krylov vectors at each step.
        This should be used if `neig>1`.

    Args:
        jax: The `jax` module.
    Returns:
        Callable: A jitted function that does a lanczos iteration.

    """

    @functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
    def jax_lanczos(matvec: Callable, arguments: List, init: jax.Array,
                                    ncv: int, neig: int, landelta: float, reortho: bool,
                                    precision: jax.lax.Precision) -> Tuple[jax.Array, List]:
        """
        Lanczos iteration for symmeric eigenvalue problems. If reortho = False,
        the Krylov basis is constructed without explicit re-orthogonalization. 
        In infinite precision, all Krylov vectors would be orthogonal. Due to 
        finite precision arithmetic, orthogonality is usually quickly lost. 
        For reortho=True, the Krylov basis is explicitly reorthogonalized.

        Args:
            matvec: A callable implementing the matrix-vector product of a
                linear operator.
            arguments: Arguments to `matvec` additional to an input vector.
                `matvec` will be called as `matvec(init, *args)`.
            init: An initial input vector to `matvec`.
            ncv: Number of krylov iterations (i.e. dimension of the Krylov space).
            neig: Number of eigenvalue-eigenvector pairs to be computed.
            landelta: Convergence parameter: if the norm of the current Lanczos vector
                falls below `landelta`, iteration is stopped.
            reortho: If `True`, reorthogonalize all krylov vectors at each step.
                This should be used if `neig>1`.
            precision: jax.lax.Precision type used in jax.numpy.vdot

        Returns:
            jax.Array: Eigenvalues
            List: Eigenvectors
            int: Number of iterations
        """
        shape = init.shape
        dtype = init.dtype
        iterative_classical_gram_schmidt = _iterative_classical_gram_schmidt(jax)
        mask_slice = (slice(ncv + 2), ) + (None,) * len(shape)
        def scalar_product(a, b):
            i1 = list(range(len(a.shape)))
            i2 = list(range(len(b.shape)))
            return jax.numpy.tensordot(a.conj(), b, (i1, i2), precision=precision)

        def norm(a):
            return jax.numpy.sqrt(scalar_product(a, a))

        def body_lanczos(vals):
            krylov_vectors, alphas, betas, i = vals
            previous_vector = krylov_vectors[i]
            def body_while(vals):
                pv, kv, _ = vals
                pv = iterative_classical_gram_schmidt(
                        pv, (i > jax.numpy.arange(ncv + 2))[mask_slice] * kv, precision)[0]
                return [pv, kv, False]

            def cond_while(vals):
                return vals[2]

            previous_vector, krylov_vectors, _ = jax.lax.while_loop(
                    cond_while, body_while,
                    [previous_vector, krylov_vectors, reortho])

            beta = norm(previous_vector)
            normalized_vector = previous_vector / beta
            Av = matvec(normalized_vector, *arguments)
            alpha = scalar_product(normalized_vector, Av)
            alphas = alphas.at[i - 1].set(alpha)
            betas = betas.at[i].set(beta)

            def while_next(vals):
                Av, _ = vals
                res = Av - normalized_vector * alpha -   krylov_vectors[i - 1] * beta
                return [res, False]

            def cond_next(vals):
                return vals[1]

            next_vector, _ = jax.lax.while_loop(
                    cond_next, while_next,
                    [Av, jax.numpy.logical_not(reortho)])
            next_vector = jax.numpy.reshape(next_vector, shape)

            krylov_vectors = krylov_vectors.at[i].set(normalized_vector)
            krylov_vectors = krylov_vectors.at[i + 1].set(next_vector)

            return [krylov_vectors, alphas, betas, i + 1]

        def cond_fun(vals):
            betas, i = vals[-2], vals[-1]
            norm = betas[i - 1]
            return jax.lax.cond(i <= ncv, lambda x: x[0] > x[1], lambda x: False,
                                                    [norm, landelta])

        # note: ncv + 2 because the first vector is all zeros, and the
        # last is the unnormalized residual.
        krylov_vecs = jax.numpy.zeros((ncv + 2,) + shape, dtype=dtype)
        # NOTE (mganahl): initial vector is normalized inside the loop
        krylov_vecs = krylov_vecs.at[1].set(init)

        # betas are the upper and lower diagonal elements
        # of the projected linear operator
        # the first two beta-values can be discarded
        # set betas[0] to 1.0 for initialization of loop
        # betas[2] is set to the norm of the initial vector.
        betas = jax.numpy.zeros(ncv + 1, dtype=dtype)
        betas = betas.at[0].set(1.0)
        # diagonal elements of the projected linear operator
        alphas = jax.numpy.zeros(ncv, dtype=dtype)
        initvals = [krylov_vecs, alphas, betas, 1]
        krylov_vecs, alphas, betas, numits = jax.lax.while_loop(
                cond_fun, body_lanczos, initvals)
        # FIXME (mganahl): if the while_loop stopps early at iteration i, alphas
        # and betas are 0.0 at positions n >= i - 1. eigh will then wrongly give
        # degenerate eigenvalues 0.0. JAX does currently not support
        # dynamic slicing with variable slice sizes, so these beta values
        # can't be truncated. Thus, if numeig >= i - 1, jitted_lanczos returns
        # a set of spurious eigen vectors and eigen values.
        # If algebraically small EVs are desired, one can initialize `alphas` with
        # large positive values, thus pushing the spurious eigenvalues further
        # away from the desired ones (similar for algebraically large EVs)

        #FIXME: replace with eigh_banded once JAX supports it
        A_tridiag = jax.numpy.diag(alphas) + jax.numpy.diag(
                betas[2:], 1) + jax.numpy.diag(jax.numpy.conj(betas[2:]), -1)
        eigvals, U = jax.numpy.linalg.eigh(A_tridiag)
        eigvals = eigvals.astype(dtype)

        # expand eigenvectors in krylov basis
        def body_vector(i, vals):
            krv, unitary, vectors = vals
            dim = unitary.shape[1]
            n, m = jax.numpy.divmod(i, dim)
            vectors = vectors.at[n, :].set(vectors[n, :] + krv[m + 1] * unitary[m, n])
            return [krv, unitary, vectors]

        _vectors = jax.numpy.zeros((neig,) + shape, dtype=dtype)
        _, _, vectors = jax.lax.fori_loop(0, neig * (krylov_vecs.shape[0] - 1),
                                                                            body_vector,
                                                                            [krylov_vecs, U, _vectors])

        return jax.numpy.array(eigvals[0:neig]), [
                vectors[n] / norm(vectors[n]) for n in range(neig)
        ], numits

    return jax_lanczos

