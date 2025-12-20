from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage


def filestorage_to_tensor_no_tv(
    image_file: FileStorage,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    normalize: bool = False,
    batch: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    try:
        image_file.stream.seek(0)
    except Exception:
        pass

    img = Image.open(image_file.stream)
    img = ImageOps.exif_transpose(img).convert("RGB")

    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        img = img.resize(size, Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]

    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        tensor = (tensor - mean) / std

    if batch:
        tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

    if device is not None:
        tensor = tensor.to(device)

    return tensor
