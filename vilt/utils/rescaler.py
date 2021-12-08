from PIL import Image

from vilt.transforms.pixelbert import pixelbert_transform
import torch
import io
import pickle

IMAGE_SIZE = 384


def load_and_rescale(path):
    img = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    img = pixelbert_transform(IMAGE_SIZE)(img)
    torch.save(img, buffer)
    return pickle.dumps(img)
