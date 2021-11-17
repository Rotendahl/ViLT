import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os
import logging
from tqdm import tqdm
from glob import glob
import threading, queue

IMAGES_PER_BATCH = 100000
NR_READER_THREADS = 10
logging.basicConfig(level=logging.INFO)


def load_captions(caption_path, split):
    with open(os.path.join(caption_path, f'cc12_{split}.json')) as fp:
        captions = json.load(fp)
    return captions

def get_image_paths(root_path):
    image_path = os.path.join(root_path, 'images')
    return [os.path.join(image_path, imgId) for imgId in  os.listdir(image_path)]


def filter_images(captions, image_paths):
    matched = []
    for img_path in tqdm(image_paths, desc='Filtering images and captions'):
        img_id = img_path.split('/')[-1]
        if img_id in captions:
            matched.append(img_path)

    logging.info(f"""
        Filtered CC images:
            Nr Captions: {len(captions)}, Nr Images: {len(image_paths)}, matched: {len(matched)}
    """)

    random.shuffle(matched)
    return matched

def filter_already_written(image_paths, destination, split):
    written_files = [
        file
        for file in os.listdir(destination)
        if split in file and 'imgs' in file
    ]
    ids_written = []
    for file in written_files:
        with open(os.path.join(destination, file), 'r') as f:
            ids_written += f.read().split('\n')

    images_not_written = [
        img_path
        for img_path in image_paths
        if img_path.split('/')[-1] not in ids_written
    ]
    if len(ids_written) > 0:
        logging.info(f"""
            Already written: {len(ids_written)},
            To Go: {len(images_not_written)}
        """)
    return images_not_written



def create_work_batches_indices(image_paths, IMAGES_PER_BATCH=IMAGES_PER_BATCH):
    if len(image_paths) <= IMAGES_PER_BATCH:
        return [(0, len(image_paths))]

    nr_batches = int(len(image_paths) // IMAGES_PER_BATCH)
    return [
        (i * IMAGES_PER_BATCH, (i + 1) * IMAGES_PER_BATCH)
        for i in range(nr_batches)
    ]

def path2rest(path, captions, split):
    img_id = path.split('/')[-1]

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

def image_reader(image_queue,captions, split, errs, imgs_written, table):
    while True:
        path = image_queue.get()
        img_id = path.split('/')[-1]
        image_data = path2rest(path, captions, split)
        if image_data is None:
            errs.put(img_id)
        else:
            imgs_written.put(img_id)
            table.put(image_data)

        image_queue.task_done()

def handle_batch(captions, image_paths, split, destination, batch_nr):
    image_queue = queue.Queue()
    table_batch = queue.Queue()
    imgs_written = queue.Queue()
    errs = queue.Queue()
    for img_path in image_paths:
        image_queue.put(img_path)

    for i in range(NR_READER_THREADS):
        threading.Thread(
            target=image_reader,
            daemon=True,
            args=(image_queue, captions, split, errs, imgs_written, table_batch)
        ).start()


    image_queue.join()


    table = []
    while table_batch.qsize() > 0:
        table.append(table_batch.get())

    errors = []
    while errs.qsize() > 0:
        errors.append(errs.get())

    written = []
    while imgs_written.qsize() > 0:
        written.append(imgs_written.get())


    table = pa.Table.from_pandas(pd.DataFrame(
        table, columns=["image", "caption", "image_id", "split"],
    ))

    os.makedirs(destination, exist_ok=True)
    batch_files = [file for file in os.listdir(destination) if split in file]
    batch_file_number = 0
    if len(batch_files) > 0:
        batch_file_number = max(
            [int(file.split('_')[-1].split('.')[0]) for file in batch_files]) + 1
    with pa.OSFile(
        f"{destination}/cc12_{split}_{batch_file_number}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    if len(errors) > 0:
        with open(os.path.join(destination, f"{split}_split_{batch_nr}_errors.txt"), "w") as fp:
            fp.write("\n".join(errors))
    with open(os.path.join(destination, f"{split}_imgs_split_{batch_nr}.txt"), "w") as fp:
        fp.write("\n".join(written))


def handle_split(caption_root, image_root, split, destination, images_per_batch):
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    captions = load_captions(caption_root, split)
    image_paths = get_image_paths(image_root)
    image_paths = filter_images(captions, image_paths)
    image_paths = filter_already_written(image_paths, destination, split)

    work_batches_indices = create_work_batches_indices(image_paths, IMAGES_PER_BATCH=images_per_batch)
    for batch_nr in tqdm(range(len(work_batches_indices))):
        logging.info(f"Handling Batch: {batch_nr+1} of {len(work_batches_indices)}")
        (batch_i, batch_j) = work_batches_indices[batch_nr]
        batch_paths = image_paths[batch_i:batch_j]
        batch_captions = [
            captions[img_path.split('/')[-1]] for img_path in batch_paths
        ]
        handle_batch(
            batch_captions,
            batch_paths,
            split,
            destination,
            batch_nr
        )


def make_arrow(caption_root,image_root, dataset_root, IMAGES_PER_BATCH=100000) :
    handle_split(caption_root, image_root, 'val', dataset_root, images_per_batch=IMAGES_PER_BATCH)
    logging.info(f"""Finished Validation images! """)
    handle_split(caption_root, image_root, 'train', dataset_root, images_per_batch=IMAGES_PER_BATCH)
    logging.info(f"""Finished Training images! """)
