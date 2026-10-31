"""TensorFlow setup and inference for the frozen flow classifier."""

from __future__ import annotations

import os


_runtime_device: str | None = None


def _tensorflow():
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return tf


def prepare_runtime(seed: int, device: str) -> None:
    global _runtime_device
    if _runtime_device is not None:
        if _runtime_device != device:
            raise RuntimeError("Changing TensorFlow device in one process is unsupported")
        tf = _tensorflow()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        return
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf = _tensorflow()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_NUM_INTRAOP_THREADS"]))
        tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_NUM_INTEROP_THREADS"]))
    except RuntimeError as error:
        raise RuntimeError("TensorFlow initialized before thread configuration") from error
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("model.device is gpu, but TensorFlow found no usable GPU")
    if device != "cpu" and gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        tf.config.set_visible_devices([], "GPU")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    _runtime_device = device

