import torch

configs = {
    "batch_size": 4,
    "lr": 1e-4,
    "n_epochs": 50,
    "max_seq_len": 128,
    "tokenizer": "bert-base-cased",
    "model_path": "./model_image_captioning_eff_transfomer.pt",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "embedding_dim": 512,
    "attention_dim": 256,
    "num_layers": 8,
    "num_heads": 8,
    "dropout": 0.1,
    "image_dir:": "../coco/",
    "karpathy_json_path": "../coco/karpathy/dataset_coco.json",
}