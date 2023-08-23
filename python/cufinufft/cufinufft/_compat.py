import inspect

import numpy as np


def get_array_ptr(data):
    try:
        return data.__cuda_array_interface__['data'][0]
    except RuntimeError:
        # Handle torch with gradient enabled
        # https://github.com/flatironinstitute/finufft/pull/326#issuecomment-1652212770
        return data.data_ptr()
    except AttributeError:
        raise TypeError("Invalid GPU array implementation. Implementation must implement the standard cuda array interface.")


def get_array_module(obj):
    module_name = inspect.getmodule(type(obj)).__name__

    if module_name.startswith("numba.cuda"):
        return "numba"
    elif module_name.startswith("torch"):
        return "torch"
    elif module_name.startswith("pycuda"):
        return "pycuda"
    else:
        return "generic"


def get_array_size(obj):
    array_module = get_array_module(obj)

    if array_module == "torch":
        return len(obj)
    else:
        return obj.size


def get_array_dtype(obj):
    array_module = get_array_module(obj)

    if array_module == "torch":
        dtype_str = str(obj.dtype)
        dtype_str = dtype_str[len("torch."):]
        return np.dtype(dtype_str)
    else:
        return obj.dtype


def is_array_contiguous(obj):
    array_module = get_array_module(obj)

    if array_module == "numba":
        return obj.is_c_contiguous()
    elif array_module == "torch":
        return obj.is_contiguous()
    else:
        return obj.flags.c_contiguous


def array_can_contiguous(obj):
    array_module = get_array_module(obj)

    if array_module == "pycuda":
        return False
    else:
        return True


def array_contiguous(obj):
    array_module = get_array_module(obj)

    if array_module == "numba":
        import numba
        ret = numba.cuda.device_array(obj.shape, obj.dtype, stream=obj.stream)
        ret[:] = obj[:]
        return ret
    if array_module == "torch":
        return obj.contiguous()
    else:
        return obj.copy(order="C")


def array_empty_like(obj, *args, **kwargs):
    module_name = get_array_module(obj)

    if module_name == "numba":
        import numba.cuda
        return numba.cuda.device_array(*args, **kwargs)
    elif module_name == "torch":
        import torch
        if "shape" in kwargs:
            kwargs["size"] = kwargs.pop("shape")
        if "dtype" in kwargs:
            dtype = kwargs.pop("dtype")
            if dtype == np.complex64:
                dtype = torch.complex64
            elif dtype == np.complex128:
                dtype = torch.complex128
            kwargs["dtype"] = dtype
        if "device" not in kwargs:
            kwargs["device"] = obj.device

        return torch.empty(*args, **kwargs)
    else:
        return type(obj)(*args, **kwargs)
