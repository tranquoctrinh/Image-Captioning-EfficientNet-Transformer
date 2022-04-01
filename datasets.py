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
                if self.phase == "train" and image["split"] in {"train", "restval"}:
                    df.append({"image_path": image_path, "caption": caption, "all_captions": captions+[""]*(10-len(captions))})
                elif self.phase == "val" and image["split"] in {"val"}:
                    df.append({"image_path": image_path, "caption": caption, "all_captions": captions+[""]*(10-len(captions))})
                elif self.phase == "test" and image["split"] in {"test"}:
                    df.append({"image_path": image_path, "caption": caption, "all_captions": captions+[""]*(10-len(captions))})
        return pd.DataFrame(df).sample(frac=0.0001).reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = self.df.iloc[index]["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        caption = self.df.loc[index, "caption"]
        caption_tokens = self.tokenizer(caption, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"][0]
        all_captions = self.df.loc[index, "all_captions"]
        all_captions_tokens = self.tokenizer(all_captions, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        return {
            "image": image,
            "caption_seq": caption,
            "caption": caption_tokens,
            "all_captions_seq": all_captions,
            "all_captions": all_captions_tokens
        }

# Test
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = ImageCaptionDataset(
        karpathy_json_path="../coco/dataset_coco.json", 
        image_dir="./coco/coco_images/", 
        tokenizer=tokenizer,
        max_seq_len=128,
        transform=transform, 
        phase="train"
    )
    print(dataset[0])