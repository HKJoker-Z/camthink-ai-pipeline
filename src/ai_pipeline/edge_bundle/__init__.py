"""Edge runtime bundle shipped to deployment targets.

``camthink_edge`` is a standalone Python 3.10 package (stdlib + numpy + the target's installed
TensorRT bindings). It is **data** from the control plane's point of view: nothing in
``ai_pipeline`` imports it; targets copy the directory to the device and run it there.
"""
