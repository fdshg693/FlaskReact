"""Utilities for converting Flask/Werkzeug uploads into PyTorch tensors.

This module intentionally avoids torchvision dependencies.
"""

from contextlib import suppress
from typing import Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage

from config import PROJECTPATHS


def filestorage_to_tensor_no_tv(
    image_file: FileStorage,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    normalize: bool = False,
    batch: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convert an uploaded image file into a float tensor.

    Designed for Flask endpoints receiving `werkzeug.datastructures.FileStorage`.
    The output tensor is scaled to [0, 1] and formatted as CHW (or NCHW if batched).

    Args:
        image_file: File uploaded via Flask/Werkzeug.
        size: If given, resizes the image. `int` means square (size, size),
            otherwise (width, height).
        normalize: If True, applies ImageNet-style normalization using
            mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        batch: If True, prepends a batch dimension (N=1).
        device: If given, moves the returned tensor to this device.

    Returns:
        A torch.FloatTensor of shape [3, H, W] or [1, 3, H, W].

    Notes:
        - Uses EXIF orientation (`ImageOps.exif_transpose`) to avoid rotated outputs.
        - Keeps everything in float32.
        - Avoids torchvision transforms (hence the "no_tv" suffix).
    """

    # Some servers/middlewares may have already read the stream; rewind if possible.
    with suppress(Exception):
        image_file.stream.seek(0)

    # Decode image from the upload stream, fix orientation, and standardize to RGB.
    # (Many ML models assume 3-channel RGB input.)
    img = Image.open(image_file.stream)
    img = ImageOps.exif_transpose(img).convert("RGB")

    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        # Pillow 10+ deprecates Image.BILINEAR in favor of Image.Resampling.BILINEAR.
        resampling = getattr(Image, "Resampling", Image)
        bilinear = getattr(resampling, "BILINEAR", 2)
        img = img.resize(size, resample=bilinear)

    # Convert PIL -> NumPy -> torch.Tensor
    # NumPy gives HWC; PyTorch convention is CHW.
    arr: NDArray[np.float32] = (
        np.asarray(img, dtype=np.float32) / 255.0
    )  # [H, W, 3] in [0, 1]
    tensor = torch.as_tensor(arr).permute(2, 0, 1).contiguous()  # [3, H, W]

    if batch:
        tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

    # Move to the target device before normalization to avoid extra transfers.
    if device is not None:
        tensor = tensor.to(device)

    if normalize:
        # Create mean/std on the same device/dtype as `tensor` to avoid warnings
        # and keep broadcast behavior consistent for both CHW and NCHW.
        if tensor.dim() == 4:
            mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        else:
            mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

    return tensor


if __name__ == "__main__":
    import mimetypes
    import sys
    from io import BytesIO

    path = PROJECTPATHS.llm_data / "image" / "sample_fish.png"
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        sys.exit(2)

    try:
        data = path.read_bytes()
    except OSError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(2)

    content_type, _ = mimetypes.guess_type(str(path))
    fs = FileStorage(
        stream=BytesIO(data),
        filename=path.name,
        name="file",
        content_type=content_type,
    )

    try:
        tensor = filestorage_to_tensor_no_tv(
            fs,
            size=224,
        )
    except Exception as e:
        print(f"Error converting image: {e}", file=sys.stderr)
        sys.exit(1)

    t_min = tensor.min().item()
    t_max = tensor.max().item()
    print(
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"min={t_min:.6g} max={t_max:.6g}"
    )
