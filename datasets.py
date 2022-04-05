import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import BertTokenizer
import json
from tqdm import tqdm

from utils import transform


def create_image_inputs(karpathy_json_path, image_dir, transform):
    """
    This function preprocesses the image and saves it in the image_dir.
    I's faster for the training process to load the images from the torch file.
    """
    karpathy = json.load(open(karpathy_json_path, "r"))
    bar = tqdm(karpathy["images"])
    for image in bar:
        image_path = os.path.join(image_dir, image["filepath"], image["filename"])
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        torch.save(image, image_path.replace(".jpg", ".pt"))


class ImageCaptionDataset(Dataset):
    def __init__(self, karpathy_json_path, image_dir, tokenizer, max_seq_len=256, transform=None, phase="train"):
        self.transform = transform
        self.tokenizer = tokenizer
        self.karpathy_json_path = karpathy_json_path
        self.image_dir = image_dir
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.df = self.create_inputs()
    
    def create_inputs(self):
        df = []
        data = json.load(open(self.karpathy_json_path, "r"))
        for image in data["images"]:
            image_path = os.path.join(self.image_dir, image["filepath"], image["filename"])
            captions = [" ".join(c["tokens"]) for c in image["sentences"]]
            for caption in captions:
                row = {
                    "image_id": image["cocoid"],
                    "image_path": image_path, 
                    "caption": caption, 
                    "all_captions": captions+[""]*(10-len(captions))
                    }
                if self.phase == "train" and image["split"] in {"train", "restval"}:
                    df.append(row)
                elif self.phase == "val" and image["split"] in {"val"}:
                    df.append(row)
                elif self.phase == "test" and image["split"] in {"test"}:
                    df.append(row)
        return pd.DataFrame(df)#.sample(frac=0.0001).reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = self.df.iloc[index]["image_path"]
        image_torch_path = image_path.replace(".jpg", ".pt")
        if os.path.exists(image_torch_path):
            image = torch.load(image_torch_path)
        else:
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            torch.save(image, image_torch_path)

        caption = self.df.loc[index, "caption"]
        caption_tokens = self.tokenizer(caption, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"][0]
        all_captions = self.df.loc[index, "all_captions"]
        all_captions_tokens = self.tokenizer(all_captions, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        return {
            "image_id": self.df.loc[index, "image_id"],
            "image_path": image_path,
            "image": image,
            "caption_seq": caption,
            "caption": caption_tokens,
            "all_captions_seq": all_captions,
            "all_captions": all_captions_tokens
        }

# Test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased", help="Bert tokenizer")
    parser.add_argument("--max_seq_len", "-msl", type=int, default=128, help="Maximum sequence length for caption generation")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size")
    # Data parameters
    parser.add_argument("--image_dir", "-id", type=str, default="../coco/", help="Path to image directory, this contains train2014, val2014")
    parser.add_argument("--karpathy_json_path", "-kap", type=str, default="../coco/karpathy/dataset_coco.json", help="Path to karpathy json file")
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    dataset = ImageCaptionDataset(
        karpathy_json_path=args.karpathy_json_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform=transform, 
        phase="train"
    )
    print(dataset[0])

    create_image_inputs(args.karpathy_json_path, args.image_dir, transform)