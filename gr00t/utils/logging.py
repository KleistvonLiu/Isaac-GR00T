from typing import Any, Dict, Sequence
import torch
import numpy as np

def _format_shape(obj: Any) -> str:
    """返回对象的形状（或类型）描述字符串。"""
    if isinstance(obj, torch.Tensor):
        return f"Tensor{tuple(obj.shape)}"
    if isinstance(obj, np.ndarray):
        return f"ndarray{obj.shape}"
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        # 处理 tuple / list：展示其中元素的形状或类型
        inner = ", ".join(_format_shape(x) for x in obj)
        return f"tuple({inner})"
    return type(obj).__name__

def print_dict_shapes(d: Dict[str, Any], indent: int = 0) -> None:
    """递归打印字典中每个键及其值的形状/类型。"""
    pad = " " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{pad}{k}:")
                print_dict_shapes(v, indent + 2)
            else:
                print(f"{pad}{k}: {_format_shape(v)}")
    else:
        print(f"{pad}: {_format_shape(d)}")   # 非 dict 直接打印
    exit(1)