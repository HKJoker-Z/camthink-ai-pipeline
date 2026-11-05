"""camthink_edge: lightweight, configuration-driven TensorRT inference runtime for Jetson.

Independent of any vendor application. Model specifics (classes, input size, thresholds,
pre/post-processing) come from a ``runtime.json`` written per deployment, so different
object-detection models reuse this code unchanged.

Dependencies on the device: Python >= 3.8, numpy, the JetPack TensorRT Python bindings,
libcudart (via ctypes), and OpenCV *or* Pillow for image IO. RTSP monitoring and the live-view
server (``serve`` with ``service/app.json`` sources, ``liveview``) need OpenCV; hardware
decoding uses OpenCV's GStreamer backend with ``nvv4l2decoder`` when available.
"""

VERSION = "0.3.0"
