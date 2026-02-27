import cv2
import numpy as np
from patchify import patchify

cf = {
    "image_size": 256,
    "num_channels": 3,
    "patch_size": 16,
}
cf["num_patches"] = (cf["image_size"]**2) // (cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"] * cf["patch_size"] * cf["num_channels"]
)

def load_and_preprocess_image_from_bytes(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    norm_image = image / 255.0
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(norm_image, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"]).astype(np.float32)
    patches = np.expand_dims(patches, axis=0)
    return image, patches

def overlay_mask_on_image(image, mask, alpha=0.5):
    image = (image * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 1] = mask
    return cv2.addWeighted(image, 1.0, mask_colored, alpha, 0)
