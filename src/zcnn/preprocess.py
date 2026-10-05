"""Turn an arbitrary photo into the 32x32 RGB uint8 tensor the model consumes."""
import numpy as np
from PIL import Image, ImageOps

from . import config as C


def center_crop_resize(img: Image.Image, size=C.IMG_SIZE):
    """Center square crop, then bilinear resize. Returns HxWx3 uint8."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(img.resize((size, size), Image.BILINEAR))
