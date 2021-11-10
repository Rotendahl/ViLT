import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob

image_path = f"/science/image/nlp-datasets/emanuele/data/cc12m/images/"

def path2rest(file_path, all_captions, split):
    with open(file_path, "rb") as fp:
        binary = fp.read()

    file_name = file_path.split('/')[-1]
    captions = all_captions[file_name]

    return [
        binary,
        captions,
        file_name.split('.')[0],
        split,
    ]

def make_arrow(root, dataset_root):
    root ='/science/image/nlp-datasets/emanuele/data/cc12m/cc12m.json'
    captions_path = root
    with open(captions_path) as fp:
        iid2captions = json.load(fp)

    image_file_ids = os.listdir(image_path)
    caption_paths = [
        os.path.join(image_path, imgId)
        for imgId in  image_file_ids
        if imgId in iid2captions.keys()
    ]
    random.shuffle(image_file_ids)
    if len(image_file_ids) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        f"Nr Image file Ids: {len(image_file_ids)}, Nr Captions {len(iid2captions)}, \
            matched = {len(caption_paths)}"
    )
    val_paths = caption_paths[:50_000]
    train_paths = caption_paths[50_000:]

    for split in ["val", "train"]:
        paths = val_paths if split == 'val' else train_paths
        sub_len = int(len(paths) // 100000)
        subs = list(range(sub_len + 1))
        for sub in subs:
            sub_paths = paths[sub * 100000 : (sub + 1) * 100000]
            bs = [path2rest(path, iid2captions, split) for path in tqdm(sub_paths)]
            dataframe = pd.DataFrame(
                bs, columns=["image", "caption", "image_id", "split"],
            )

            table = pa.Table.from_pandas(dataframe)

            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                f"{dataset_root}/cc12_{split}_{sub}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del bs
            gc.collect()
