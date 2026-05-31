"""Parameter initialization methods (Xavier, He)."""

import numpy as np


def _calculate_fan_in_fan_out(tensor_data: np.ndarray) -> tuple:
    """Compute fan_in and fan_out from weight shape (out_features, in_features)."""
    if tensor_data.ndim < 2:
        fan_in = fan_out = tensor_data.shape[0]
    else:
        fan_in = tensor_data.shape[1]   # in_features
        fan_out = tensor_data.shape[0]  # out_features
    return fan_in, fan_out


def xavier_uniform_(tensor):
    """Xavier uniform initialization (Glorot uniform).

    Suitable for Sigmoid / Tanh activations.
    W ~ U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
    """
    fan_in, fan_out = _calculate_fan_in_fan_out(tensor.data)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    tensor.data = np.random.uniform(-limit, limit, tensor.data.shape).astype(np.float64)


def xavier_normal_(tensor):
    """Xavier normal initialization (Glorot normal).

    Suitable for Sigmoid / Tanh activations.
    W ~ N(0, sqrt(2/(fan_in+fan_out)))
    """
    fan_in, fan_out = _calculate_fan_in_fan_out(tensor.data)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data = np.random.normal(0, std, tensor.data.shape).astype(np.float64)


def he_uniform_(tensor):
    """He uniform initialization (Kaiming uniform).

    Suitable for ReLU activations.
    W ~ U[-sqrt(6/fan_in), sqrt(6/fan_in)]
    """
    fan_in, _ = _calculate_fan_in_fan_out(tensor.data)
    limit = np.sqrt(6.0 / fan_in)
    tensor.data = np.random.uniform(-limit, limit, tensor.data.shape).astype(np.float64)


def he_normal_(tensor):
    """He normal initialization (Kaiming normal).

    Suitable for ReLU activations.
    W ~ N(0, sqrt(2/fan_in))
    """
    fan_in, _ = _calculate_fan_in_fan_out(tensor.data)
    std = np.sqrt(2.0 / fan_in)
    tensor.data = np.random.normal(0, std, tensor.data.shape).astype(np.float64)
