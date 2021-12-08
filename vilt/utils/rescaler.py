from PIL import Image

from vilt.transforms.pixelbert import pixelbert_transform

IMAGE_SIZE = 384


def load_and_rescale(path):
    img = Image.open(path).convert("RGB")
    return pixelbert_transform(IMAGE_SIZE)(img).tolist()
