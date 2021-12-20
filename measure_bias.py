import torch
from os import path
import copy
import time
import requests
import os
import io
import numpy as np
import re
import spacy
import ipdb
import pandas as pd
import pickle
import os
from collections import defaultdict
import argparse
import pandas as pd
from tqdm import tqdm
import spacy
import logging
from time import sleep
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from vilt.modules.objectives import cost_matrix_cosine, ipot
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer

tokenizer = None
model = None
device = None


class ViltBias:
    def __init__(self, gender_file, bias_file, pickled_dir, dest_file, model) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.dest_file = dest_file
        self.pickled_dir = pickled_dir

        bias_tokens = pd.read_csv(bias_file)["tokens"].dropna().to_list()
        bias_tokens = " ".join(bias_tokens).lower()
        self.bias_tokens = [t.lemma_ for t in self.nlp(bias_tokens)]

        gender_tokens = pd.read_csv(gender_file, sep=";")
        self.male_tokens = gender_tokens["Male"].dropna().to_list()
        self.female_tokens = gender_tokens["Female"].dropna().to_list()

        self.male_tokens_ids = tokenizer.convert_tokens_to_ids(self.male_tokens)
        self.female_tokens_ids = tokenizer.convert_tokens_to_ids(self.female_tokens)

        self.male_lemmas = [
            t.lemma_ for t in self.nlp(" ".join(self.male_tokens).lower())
        ]
        self.female_lemmas = [
            t.lemma_ for t in self.nlp(" ".join(self.female_tokens).lower())
        ]

    def is_male_token(self, file_path, mp_text):
        try:
            image = Image.open(file_path).convert("RGB")
            img = pixelbert_transform(size=384)(image)
            img = img.unsqueeze(0).to(device)
        except:
            return None

        batch = {"text": [""], "image": [None]}
        tl = len(re.findall("\[MASK\]", mp_text))
        inferred_token = [mp_text]
        batch["image"][0] = img

        with torch.no_grad():
            for i in range(tl):
                encoded = tokenizer(inferred_token)
                inferred_token = [
                    tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(encoded["input_ids"][0][:40])
                    )
                ]
                batch["text"] = inferred_token
                batch["text_ids"] = torch.tensor([encoded["input_ids"][0][:40]]).to(
                    device
                )
                batch["text_labels"] = torch.tensor([encoded["input_ids"][0][:40]]).to(
                    device
                )
                batch["text_masks"] = torch.tensor(
                    [encoded["attention_mask"][0][:40]]
                ).to(device)
                encoded = encoded["input_ids"][0][1:40]
                infer = model(batch)
                mlm_logits = model.mlm_score(infer["text_feats"])[0, 1:-1]
                has_mask = (
                    tokenizer.convert_tokens_to_ids(tokenizer.mask_token) in encoded
                )
                mask_index = 0
                if has_mask:
                    mask_index = encoded.index(
                        tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                    )
                if has_mask and mask_index < mlm_logits.size()[0]:
                    token_props = torch.softmax(mlm_logits, dim=1)[mask_index]
                    male_props = token_props[self.male_tokens_ids].sum()
                    female_props = token_props[self.female_tokens_ids].sum()
                    return male_props > female_props
                else:
                    return None

    def get_image_path(self, image_id, split, dataset):
        if dataset == "VG":
            base_path = "/home/xmt224/data/image_data/VG/images"
            VG1 = path.join(base_path, "VG_100K", f"{image_id}.jpg")
            VG2 = path.join(base_path, "VG_100K_2", f"{image_id}.jpg")
            return VG1 if path.exists(VG1) else VG2
        if dataset == "cc":
            base_path = "/science/image/nlp-datasets/emanuele/data/cc12m/images"
            return path.join(base_path, image_id)
        if dataset == "gcc":
            sub_dir = "training" if split == "train" else "validation"
            base_path = path.join(
                "/science/image/nlp-datasets/emanuele/data/conceptual_captions/all/",
                sub_dir,
                "clean",
            )
            return path.join(base_path, image_id)
        if dataset == "coco":
            base_path = "/science/image/nlp-datasets/emanuele/data/mscoco/images"
            sub_dir = "train2014" if "train" in image_id else "val2014"
            image_id = image_id.split("_")[-1]
            return path.join(base_path, sub_dir, image_id)
        else:
            res = "Unsupported dataset!"

    def bias_metrics_for_file(self, caption_path):
        with open(caption_path, "rb") as f:
            captions = pickle.load(f)

        file_paths = []
        for i, row in captions.iterrows():
            file_paths.append(self.get_image_path(row.img_id, row.split, row.dataset))
        captions["file_path"] = file_paths

        bias_counts = defaultdict(lambda: {"male": 0, "female": 0})
        for i, row in tqdm(captions.iterrows(), desc=f"handling file {caption_path}"):
            caption = row["tokens"]
            raw_caption = ""
            nsubj = None
            has_mask = False
            for token in caption:
                if token.dep_ == "nsubj" and not has_mask:
                    nsubj = token.lemma_.lower()
                    raw_caption += "[MASK]"
                    has_mask = True
                else:
                    raw_caption += str(token)
                raw_caption += token.whitespace_

            if nsubj is not None:
                captions_lemmas = [t.lemma_.lower() for t in caption]
                for bias_lemma in self.bias_tokens:
                    if bias_lemma in captions_lemmas:
                        is_male = self.is_male_token(row.file_path, raw_caption)
                        if is_male is None:
                            continue
                        elif is_male:
                            bias_counts[bias_lemma]["male"] += 1
                        else:
                            bias_counts[bias_lemma]["female"] += 1
        return bias_counts

    def handle_files(self):
        files_to_filter = [
            os.path.join(os.path.join(self.pickled_dir), f)
            for f in os.listdir(self.pickled_dir)
            if "pickle" in f
        ]
        to_merge = []
        for file in tqdm(files_to_filter, desc="Measuring Bias"):
            to_merge.append(self.bias_metrics_for_file(file))

        (out_dir, _) = os.path.split(self.dest_file)
        os.makedirs(out_dir, exist_ok=True)

        global_bias_counts = defaultdict(lambda: {"male": 0, "female": 0})
        for file_counts in tqdm(to_merge, desc="Merging gender counts"):
            roots = file_counts.keys()
            for root in roots:
                global_bias_counts[root]["male"] += file_counts[root]["male"]
                global_bias_counts[root]["female"] += file_counts[root]["female"]

        df = pd.DataFrame(global_bias_counts).transpose()
        df.to_csv(self.dest_file)
        return df


@ex.automain
def main(_config):
    global tokenizer
    global model
    global device
    _config = copy.deepcopy(_config)
    loss_names = {
        "itm": 0,
        "mlm": 0.5,
        "mpp": 0,
        "vqa": 0,
        "imgcls": 0,
        "nlvr2": 0,
        "irtr": 0,
        "arc": 0,
    }
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    _config.update(
        {
            "loss_names": loss_names,
        }
    )
    model = ViLTransformerSS(_config)
    model.setup("test")
    model.eval()
    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)
    viltBias = ViltBias(
        "/home/xmt224/assets/Mappings.csv",
        "/home/xmt224/code/utils/bias/bias_tokens.csv",
        "/home/xmt224/data/captions/gender_filtered",
        "/home/xmt224/data/bias-all/vilt_mapped15_bias.csv",
        model,
    )
    viltBias.handle_files()
