import torch
import matplotlib.pyplot as plt
import os


configs = {
    "batch_size": 16,
    "lr": 1e-4,
    "n_epochs": 50,
    "max_seq_len": 128,
    "tokenizer": "bert-base-cased",
    "model_path": "./model_image_captioning_eff_transfomer.pt",
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    "embedding_dim": 512,
    "attention_dim": 256,
    "num_layers": 8,
    "num_heads": 8,
    "dropout": 0.1,
    "early_stopping": 5,
    "image_dir": "../coco/",
    "karpathy_json_path": "../coco/karpathy/dataset_coco.json",
    "log_path": "./images/log_training.json",
    "log_visualize_dir": "./images/",
}

def visualize_log(log, configs):
    # Plot loss per epoch
    plt.figure()
    plt.plot(log["train_loss"], label="train")
    plt.plot(log["val_loss"], label="val")
    plt.legend()
    plt.title("Loss per epoch")
    filename = os.path.join(configs["log_visualize_dir"], "loss_epoch.png")
    plt.savefig(filename)

    # Plot bleu4 per epoch
    plt.figure()
    plt.plot(log["train_bleu4"], label="train")
    plt.plot(log["val_bleu4"], label="val")
    plt.legend()
    plt.title("BLEU-4 per epoch")
    filename = os.path.join(configs['log_visualize_dir'], 'bleu4_epoch.png')
    plt.savefig(filename)

    # Plot loss per batch
    plt.figure()
    train_loss_batch = []
    for loss in log["train_loss_batch"]:
        train_loss_batch += loss
    plt.plot(train_loss_batch, label="train")
    
    val_loss_batch = []
    for loss in log["val_loss_batch"]:
        val_loss_batch += loss
    plt.plot(val_loss_batch, label="val")
    plt.legend()
    plt.title("Loss per batch")
    filename = os.path.join(configs['log_visualize_dir'], 'loss_batch.png')
    plt.savefig(filename)