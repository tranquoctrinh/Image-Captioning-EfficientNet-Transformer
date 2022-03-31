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
    def __init__(self, karpathy_json_path, image_folder, max_seq_len=256, transform=None, phase="train"):
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.karpathy_json_path = karpathy_json_path
        self.image_folder = image_folder
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.df = self.create_inputs()
    
    def create_inputs(self):
        df = []
        data = json.load(open(self.karpathy_json_path, "r"))
        for image in data["images"]:
            image_path = os.path.join(self.image_folder, image["filepath"], image["filename"])
            for c in image["sentences"]:
                caption = " ".join(c["tokens"])
                if self.phase == "train" and image["split"] in {"train", "restval"}:
                    df.append({"image_path": image_path, "caption": caption})
                elif self.phase == "val" and image["split"] in {"val"}:
                    df.append({"image_path": image_path, "caption": caption})
                elif self.phase == "test" and image["split"] in {"test"}:
                    df.append({"image_path": image_path, "caption": caption})
        return pd.DataFrame(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # image_path = self.df.iloc[index]["image_path"]
        # image = Image.open(image_path).convert("RGB")
        # random a image by numpy
        image = np.random.rand(224, 224, 3)
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        print(image.shape)
        caption = self.df.iloc[index]["caption"]
        tokens = self.tokenizer(caption, max_length=self.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"][0]
        import ipdb; ipdb.set_trace()
        return image, torch.LongTensor(tokens)

# Test
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageCaptionDataset(
        karpathy_json_path="../coco/dataset_coco.json", 
        image_folder="./coco/coco_images/", 
        max_seq_len=128,
        transform=transform, 
        phase="test"
    )
    dataset[0]