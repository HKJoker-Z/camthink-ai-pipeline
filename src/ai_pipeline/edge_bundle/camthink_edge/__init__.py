"""camthink_edge: lightweight, configuration-driven TensorRT inference runtime for Jetson.

Independent of any vendor application. Model specifics (classes, input size, thresholds,
pre/post-processing) come from a ``runtime.json`` written per deployment, so different
object-detection models reuse this code unchanged.

Dependencies on the device: Python >= 3.10, numpy, the JetPack TensorRT Python bindings,
libcudart (via ctypes), and OpenCV *or* Pillow for image IO.
"""

VERSION = "0.2.0"
