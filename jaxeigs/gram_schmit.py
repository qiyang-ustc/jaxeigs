import jax
import typing
import types

__all__ = ["_iterative_classical_gram_schmidt"]

def _iterative_classical_gram_schmidt(jax: types.ModuleType) -> typing.Callable:
    def iterative_classical_gram_schmidt(
            vector: jax.Array,
            krylov_vectors: jax.Array,
            precision: jax.lax.Precision,
            iterations: int = 2,
            ) -> jax.Array:
        """
        Orthogonalize `vector`  to all rows of `krylov_vectors`.

        Args:
            vector: Initial vector.
            krylov_vectors: Matrix of krylov vectors, each row is treated as a
                vector.
            iterations: Number of iterations.

        Returns:
            jax.Array: The orthogonalized vector.
        """
        i1 = list(range(1, len(krylov_vectors.shape)))
        i2 = list(range(len(vector.shape)))

        vec = vector
        overlaps = 0
        for _ in range(iterations):
            ov = jax.numpy.tensordot(
                    krylov_vectors.conj(), vec, (i1, i2), precision=precision)
            vec = vec - jax.numpy.tensordot(
                    ov, krylov_vectors, ([0], [0]), precision=precision)
            overlaps = overlaps + ov
        return vec, overlaps

    return iterative_classical_gram_schmidt
