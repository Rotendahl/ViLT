import sys
import json
import pandas as pd
import pyarrow as pa
import os
import logging
from tqdm import tqdm
import threading, queue


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_captions(caption_path, split):
    logging.info(f"Loading {split} captions")
    with open(os.path.join(caption_path, f"cc12_{split}.json")) as fp:
        captions = json.load(fp)
    logging.info(f"Loaded {len(captions):,} captions")
    return captions


def get_image_paths(root_path):
    image_path = os.path.join(root_path, "images")
    return [
        os.path.join(image_path, imgId)
        for imgId in tqdm(os.listdir(image_path), desc="loading image paths")
    ]


def filter_images(captions, image_paths):
    matched = {}
    for img_path in tqdm(image_paths, desc="Filtering images and captions"):
        img_id = img_path.split("/")[-1]
        if img_id in captions:
            matched[img_id] = (img_id, captions[img_id], img_path)

    logging.info(
        f"""
        Filtered CC images:
            Nr Captions: {len(captions):,}
            Nr Images: {len(image_paths):,}
            Matched: {len(matched):,}
    """
    )
    return matched


def filter_already_written(cap_imgs, destination, split):
    written_files = [
        file for file in os.listdir(destination) if split in file and "imgs" in file
    ]
    ids_written = []
    for file in written_files:
        with open(os.path.join(destination, file), "r") as f:
            ids_written += f.read().split("\n")
    logging.info(f"For split {split}, images already written: {len(ids_written):,}")
    for written_id in ids_written:
        cap_imgs.pop(written_id, None)
    logging.info(f"Image to be written: {len(cap_imgs):,}")
    return cap_imgs


def create_work_batches_indices(cap_images):
    if len(cap_images) <= IMAGES_PER_BATCH:
        return [cap_images.keys()]

    nr_batches = int(len(cap_images) // IMAGES_PER_BATCH)
    ids = list(cap_images.keys())
    return [
        ids[i * IMAGES_PER_BATCH : (i + 1) * IMAGES_PER_BATCH]
        for i in range(nr_batches)
    ]


def path2rest(path, captions, split):
    img_id = path.split("/")[-1]

    try:
        with open(path, "rb") as fp:
            binary = fp.read()
    except:
        return None

    return [
        binary,
        captions,
        img_id,
        split,
    ]


def cap_img_reader(cap_img_queue, split, errs, imgs_written, table):
    while not cap_img_queue.empty():
        (img_id, caption, img_path) = cap_img_queue.get()
        try:
            with open(img_path, "rb") as fp:
                binary = fp.read()
            row = [
                binary,
                caption,
                img_id,
                split,
            ]
            imgs_written.put(img_id)
            table.put(row)
        except:
            errs.put(img_id)
        finally:
            cap_img_queue.task_done()


def handle_batch(captions, img_ids, split, destination, batch_nr):
    cap_img_queue = queue.Queue()
    table_batch = queue.Queue()
    imgs_written = queue.Queue()
    errs = queue.Queue()
    for img_id in img_ids:
        cap_img_queue.put(captions[img_id])

    for i in range(NR_READER_THREADS):
        threading.Thread(
            target=cap_img_reader,
            daemon=True,
            args=(cap_img_queue, split, errs, imgs_written, table_batch),
        ).start()

    cap_img_queue.join()

    table = []
    while table_batch.qsize() > 0:
        table.append(table_batch.get())

    errors = []
    while errs.qsize() > 0:
        errors.append(errs.get())

    written = []
    while imgs_written.qsize() > 0:
        written.append(imgs_written.get())

    table = pa.Table.from_pandas(
        pd.DataFrame(
            table,
            columns=["image", "caption", "image_id", "split"],
        )
    )

    os.makedirs(destination, exist_ok=True)
    batch_files = [file for file in os.listdir(destination) if split in file]
    batch_file_number = 0
    if len(batch_files) > 0:
        batch_file_number = (
            max([int(file.split("_")[-1].split(".")[0]) for file in batch_files]) + 1
        )
    with pa.OSFile(
        f"{destination}/cc12_{split}_{batch_file_number}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    if len(errors) > 0:
        with open(
            os.path.join(destination, f"{split}_split_{batch_file_number}_errors.txt"),
            "w",
        ) as fp:
            fp.write("\n".join(errors))
    with open(
        os.path.join(destination, f"{split}_imgs_split_{batch_file_number}.txt"), "w"
    ) as fp:
        fp.write("\n".join(written))


def handle_split(caption_root, image_root, split, destination):
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    captions = load_captions(caption_root, split)
    image_paths = get_image_paths(image_root)
    cap_images = filter_images(captions, image_paths)
    cap_images = filter_already_written(cap_images, destination, split)

    work_batches_img_ids = create_work_batches_indices(cap_images)
    for batch_nr in tqdm(range(len(work_batches_img_ids))):
        logging.info(f"Handling Batch: {batch_nr+1} of {len(work_batches_img_ids)}")
        handle_batch(
            cap_images, work_batches_img_ids[batch_nr], split, destination, batch_nr
        )


def make_arrow(caption_root, image_root, dataset_root):
    handle_split(caption_root, image_root, "val", dataset_root)
    logging.info(f"""Finished Validation images! """)
    handle_split(caption_root, image_root, "train", dataset_root)
    logging.info(f"""Finished Training images! """)


IMAGES_PER_BATCH = 10_000
NR_READER_THREADS = 10

if __name__ == "__main__":
    """
    Usage python write_cc12.py caption_path, image path, output_dir, batch_size nr_threads
    """
    caption_root = sys.argv[1]
    image_root = sys.argv[2]
    output_dir = sys.argv[3]
    if len(sys.argv) > 4:
        IMAGES_PER_BATCH = int(sys.argv[4])
    if len(sys.argv) > 5:
        NR_READER_THREADS = int(sys.argv[5])
    make_arrow(caption_root, image_root, output_dir)
