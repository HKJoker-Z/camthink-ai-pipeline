"""Minimal CUDA runtime bindings via ctypes (malloc/free/memcpy/stream).

Avoids a dependency on pycuda / cuda-python: libcudart ships with JetPack and the TensorRT
Python API only needs device pointers and a stream handle.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os

H2D = 1
D2H = 2

_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    candidates = [os.environ.get("CAMTHINK_CUDART"), ctypes.util.find_library("cudart")]
    candidates += sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"), reverse=True)
    candidates += sorted(glob.glob("/usr/lib/aarch64-linux-gnu/libcudart.so*"), reverse=True)
    # Plain sonames: dlopen honours LD_LIBRARY_PATH (user-space CUDA installs on JetPack 5),
    # which ctypes.util.find_library does not.
    candidates += ["libcudart.so.12", "libcudart.so.11.0", "libcudart.so"]
    errors = []
    for c in candidates:
        if not c:
            continue
        try:
            _lib = ctypes.CDLL(c)
            break
        except OSError as exc:
            errors.append(f"{c}: {exc}")
    if _lib is None:
        raise RuntimeError(f"libcudart not found; tried {candidates}; {errors}")
    _lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    _lib.cudaFree.argtypes = [ctypes.c_void_p]
    _lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    _lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    _lib.cudaGetErrorString.restype = ctypes.c_char_p
    _lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    return _lib


def _check(code: int, what: str) -> None:
    if code != 0:
        msg = _load().cudaGetErrorString(code).decode(errors="replace")
        raise RuntimeError(f"{what} failed: CUDA error {code} ({msg})")


def runtime_version() -> int:
    v = ctypes.c_int()
    _check(_load().cudaRuntimeGetVersion(ctypes.byref(v)), "cudaRuntimeGetVersion")
    return v.value


def malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    _check(_load().cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return ptr.value or 0


def free(ptr: int) -> None:
    if ptr:
        _check(_load().cudaFree(ptr), "cudaFree")


def memcpy(dst: int, src: int, nbytes: int, kind: int) -> None:
    _check(_load().cudaMemcpy(dst, src, nbytes, kind), "cudaMemcpy")


def stream_create() -> int:
    s = ctypes.c_void_p()
    _check(_load().cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
    return s.value or 0


def stream_synchronize(stream: int) -> None:
    _check(_load().cudaStreamSynchronize(stream), "cudaStreamSynchronize")


def stream_destroy(stream: int) -> None:
    if stream:
        _check(_load().cudaStreamDestroy(stream), "cudaStreamDestroy")
