import jax
import numpy as np
import scipy

eig_backend = np.linalg.eig
eigh_backend = np.linalg.eigh
eigh_tridiagoal_backend = scipy.linalg.eigh_tridiagonal
schur_backend = scipy.linalg.schur


def lapack_eig(H):
        result_shape = (jax.ShapeDtypeStruct(H.shape[0:1], H.dtype),
                                        jax.ShapeDtypeStruct(H.shape, H.dtype))
        return jax.pure_callback(eig_backend, result_shape, H)
    
def lapack_eigh(H):
    result_shape = (jax.ShapeDtypeStruct(H.shape[0:1], H.dtype),
                                    jax.ShapeDtypeStruct(H.shape, H.dtype))
    return jax.pure_callback(eig_backend, result_shape, H)

def lapack_schur(H):
    return jax.pure_callback(schur_backend, (H,H), H)

def lapack_eigh_tridiagonal(d,e):
    dim = d.shape[0]
    result_shape = (jax.ShapeDtypeStruct(dim,),
                                jax.ShapeDtypeStruct(dim, dim))
    return jax.pure_callback(schur_backend, result_shape, (d,e))