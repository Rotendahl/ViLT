import pyarrow as pa
import io
import argparse
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from vilt.transforms.pixelbert import pixelbert_transform
import logging

tqdm.pandas()


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

IMAGE_SIZE = 384
scaler = pixelbert_transform(IMAGE_SIZE, normalize=False)


def read_arrow_file(arrow_file):
    table = pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file)).read_all()
    return table.to_pandas()


def rescale(bin):
    image_bytes = io.BytesIO(bin)
    image_bytes.seek(0)
    img = Image.open(image_bytes).convert("RGB")
    img = scaler(img)
    img = transforms.ToPILImage(mode="RGB")(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def write_arrow_file(table, dest):
    img_col = "image" if "image" in table.columns else "binary"
    table[img_col] = table[img_col].progress_map(rescale)
    table = pa.Table.from_pandas(table)
    with pa.OSFile(dest, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrows_dir", type=str)
    parser.add_argument("--dest_dir", type=str)
    args = parser.parse_args()

    input_dir = args.arrows_dir
    output_dir = args.dest_dir
    input_files = [f for f in os.listdir(input_dir) if "arrow" in f]
    os.makedirs(output_dir, exist_ok=True)
    for input_file in tqdm(input_files, "rescaling arrow files"):
        table = read_arrow_file(os.path.join(input_dir, input_file))
        logging.info(f"Rescaling {input_file} with: {len(table):,} rows")
        write_arrow_file(table, os.path.join(output_dir, input_file))
