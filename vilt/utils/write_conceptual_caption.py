import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob


def path2rest(path, iid2captions):
    split, _, name = path.split("/")[-3:]
    split = split.split("_")[-1]
    iid = name

    with open(path, "rb") as fp:
        binary = fp.read()

    captions = iid2captions[iid]

    return [
        binary,
        captions,
        iid,
        split,
    ]

def make_arrow(root, dataset_root):
    root ='/science/image/nlp-datasets/emanuele/data/conceptual_captions/'
    for split in ["val", "train"]:
        captions_path = f"{root}/annotations/" + (
            'caption_valid.json' if split == 'val' else 'caption_train.json'
        )
        with open(captions_path) as fp:
            iid2captions = json.load(fp)

        image_root_path = "".join([
            "/science/image/nlp-datasets/emanuele/data/conceptual_captions/all/",
            'training' if split != 'val' else 'validation',
            "/clean"
        ])
        '/science/image/nlp-datasets/emanuele/data/conceptual_captions/all/training/clean'
        image_file_ids = os.listdir(image_root_path)
        caption_paths = [
            os.path.join(image_root_path, imgId)
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

        sub_len = int(len(caption_paths) // 100000)
        subs = list(range(sub_len + 1))
        for sub in subs:
            sub_paths = caption_paths[sub * 100000 : (sub + 1) * 100000]
            bs = [path2rest(path, iid2captions) for path in tqdm(sub_paths)]
            dataframe = pd.DataFrame(
                bs, columns=["image", "caption", "image_id", "split"],
            )

            table = pa.Table.from_pandas(dataframe)

            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                f"{dataset_root}/conceptual_caption_{split}_{sub}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del bs
            gc.collect()
