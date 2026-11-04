"""TensorRT engine execution (TensorRT 10 API: named tensors, execute_async_v3)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cudart


@dataclass
class _Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    nbytes: int
    device_ptr: int
    host: np.ndarray


class TrtEngine:
    """Loads a serialized engine and runs single-batch inference.

    Usage: ``eng = TrtEngine(path); out = eng.infer(chw_float_array)``; ``eng.close()``.
    """

    def __init__(self, engine_path: str | Path, *, input_shape: tuple[int, ...] | None = None):
        import tensorrt as trt  # device-only dependency, deliberately local

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        data = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cudart.stream_create()
        self.inputs: list[_Tensor] = []
        self.outputs: list[_Tensor] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            is_input = mode == trt.TensorIOMode.INPUT
            if is_input and any(d < 0 for d in shape):
                if input_shape is None:
                    raise RuntimeError(
                        f"input {name} has dynamic shape {shape}; input_shape required"
                    )
                self.context.set_input_shape(name, input_shape)
                shape = tuple(input_shape)
            if not is_input:
                shape = tuple(self.context.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            host = np.empty(shape, dtype=dtype)
            dev = cudart.malloc(nbytes)
            self.context.set_tensor_address(name, dev)
            t = _Tensor(name, shape, dtype, nbytes, dev, host)
            (self.inputs if is_input else self.outputs).append(t)
        if len(self.inputs) != 1:
            raise RuntimeError(
                f"expected exactly one input tensor, got {[t.name for t in self.inputs]}"
            )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.inputs[0].shape

    def describe(self) -> dict:
        return {
            "tensorrt": self.trt.__version__,
            "inputs": [
                {"name": t.name, "shape": list(t.shape), "dtype": str(t.dtype)} for t in self.inputs
            ],
            "outputs": [
                {"name": t.name, "shape": list(t.shape), "dtype": str(t.dtype)}
                for t in self.outputs
            ],
        }

    def infer(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Run one inference. ``x`` must match the input shape; returns name -> output array."""
        inp = self.inputs[0]
        if tuple(x.shape) != inp.shape:
            raise ValueError(f"input shape {x.shape} != engine input {inp.shape}")
        x = np.ascontiguousarray(x, dtype=inp.dtype)
        cudart.memcpy(inp.device_ptr, x.ctypes.data, inp.nbytes, cudart.H2D)
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        cudart.stream_synchronize(self.stream)
        out: dict[str, np.ndarray] = {}
        for t in self.outputs:
            cudart.memcpy(t.host.ctypes.data, t.device_ptr, t.nbytes, cudart.D2H)
            out[t.name] = t.host.copy()
        return out

    def time_infer(self, x: np.ndarray, iterations: int, warmup: int) -> list[float]:
        """Latency in ms per iteration (H2D + execute + D2H), after ``warmup`` runs."""
        for _ in range(warmup):
            self.infer(x)
        samples = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.infer(x)
            samples.append((time.perf_counter() - t0) * 1000.0)
        return samples

    def close(self) -> None:
        for t in self.inputs + self.outputs:
            cudart.free(t.device_ptr)
            t.device_ptr = 0
        cudart.stream_destroy(self.stream)
        self.stream = 0
