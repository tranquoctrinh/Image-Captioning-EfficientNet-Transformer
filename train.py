import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
import os
import json
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction()

from utils import configs, visualize_log, transform
from datasets import ImageCaptionDataset
from models import ImageCaptionModel


def train_epoch(model, train_loader, tokenizer, criterion, optim, epoch, device):
    model.train()
    total_loss, batch_bleu4 = [], []
    hypotheses, references = [], []
    bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch+1}")
    for i, batch in bar:
        image, caption, all_caps = batch["image"].to(device), batch["caption"].to(device), batch["all_captions_seq"]
        target_input = caption[:, :-1]
        target_mask = model.make_mask(target_input)
        preds = model(image, target_input)
        optim.zero_grad()
        gold = caption[:, 1:].contiguous().view(-1)
        loss = criterion(preds.view(-1, preds.size(-1)), gold)
        loss.backward()
        optim.step()
        total_loss.append(loss.item())
    
        # Calculate BLEU-4 score
        preds = F.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1)
        preds = preds.detach().cpu().numpy()
        caps = [tokenizer.decode(cap, skip_special_tokens=True) for cap in preds]
        hypo = [cap.split() for cap in caps]
        
        batch_size = len(hypo)
        ref = []
        for i in range(batch_size):
            ri = [all_caps[j][i].split() for j in range(len(all_caps)) if all_caps[j][i]]
            ref.append(ri)
        batch_bleu4.append(corpus_bleu(ref, hypo, smoothing_function=smoothie.method4))
        hypotheses += hypo
        references += ref

        bar.set_postfix(loss=total_loss[-1], bleu4=batch_bleu4[-1])
    
    train_bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie.method4)
    train_loss = sum(total_loss) / len(total_loss)
    return train_loss, train_bleu4, total_loss
    

def validate_epoch(model, valid_loader, tokenizer, criterion, epoch, device):
    model.eval()
    total_loss, batch_bleu4 = [], []
    hypotheses, references = [], []
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validating epoch {epoch+1}")
    for i, batch in bar:
        image, caption, all_caps = batch["image"].to(device), batch["caption"].to(device), batch["all_captions_seq"]
        target_input = caption[:, :-1]
        target_mask = model.make_mask(target_input)
        preds = model(image, target_input)

        gold = caption[:, 1:].contiguous().view(-1)
        loss = criterion(preds.view(-1, preds.size(-1)), gold)
        total_loss.append(loss.item())

        preds = F.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1)
        preds = preds.detach().cpu().numpy()
        caps = [tokenizer.decode(cap, skip_special_tokens=True) for cap in preds]
        hypo = [cap.split() for cap in caps]
        
        batch_size = len(hypo)
        ref = []
        for i in range(batch_size):
            ri = [all_caps[j][i].split() for j in range(len(all_caps)) if all_caps[j][i]]
            ref.append(ri)
        batch_bleu4.append(corpus_bleu(ref, hypo, smoothing_function=smoothie.method4))
        hypotheses += hypo
        references += ref

        bar.set_postfix(loss=total_loss[-1], bleu4=batch_bleu4[-1])

    val_loss = sum(total_loss) / len(total_loss)
    val_bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie.method4)
    return val_loss, val_bleu4, total_loss

def train(model, train_loader, valid_loader, optim, criterion, n_epochs, tokenizer, device, model_path, log_path, early_stopping=5):
    model.train()
    log = {"train_loss": [], "train_bleu4": [], "train_loss_batch": [], "val_loss": [], "val_bleu4": [], "val_loss_batch": []}
    best_train_bleu4, best_val_bleu4, best_epoch = -np.Inf, -np.Inf, 1
    count_early_stopping = 0
    start_time = time.time()

    for epoch in range(n_epochs):
        train_loss, train_bleu4, train_loss_batch = train_epoch(
            model=model,
            train_loader=train_loader,
            tokenizer=tokenizer,
            optim=optim,
            criterion=criterion,
            epoch=epoch,
            device=device
        )
        val_loss, val_bleu4, val_loss_batch = validate_epoch(
            model=model,
            valid_loader=valid_loader,
            tokenizer=tokenizer,
            criterion=criterion,
            epoch=epoch,
            device=device
        )
        
        best_train_bleu4 = train_bleu4 if train_bleu4 > best_train_bleu4 else best_train_bleu4
        
        # Detect improvement and save model or early stopping and break
        if val_bleu4 > best_val_bleu4:
            best_val_bleu4 = val_bleu4
            best_epoch = epoch + 1
            # Save Model with best validation bleu4
            torch.save(model.state_dict(), model_path)
            print("-------- Detect improment and save the best model --------")
            count_early_stopping = 0
        else:
            count_early_stopping += 1
            if count_early_stopping >= early_stopping:
                print("-------- Early stopping --------")
                break
        
        # Logfile
        log["train_loss"].append(train_loss)
        log["train_bleu4"].append(train_bleu4)
        log["train_loss_batch"].append(train_loss_batch)
        log["val_loss"].append(val_loss)
        log["val_bleu4"].append(val_bleu4)
        log["val_loss_batch"].append(val_loss_batch)
        log["best_train_bleu4"] = best_train_bleu4
        log["best_val_bleu4"] = best_val_bleu4
        log["best_epoch"] = best_epoch
        # Save log
        with open(log_path, "w") as f:
            json.dump(log, f)

        torch.cuda.empty_cache()
        
        print(f"---- Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.5f} | Valid Loss: {val_loss:.5f} | Validation BLEU-4: {val_bleu4:.5f} | Best BLEU-4: {best_val_bleu4:.5f} | Best Epoch: {best_epoch} | Time taken: {timedelta(seconds=int(time.time()-start_time))}")
    
    return log


def main():    
    device = torch.device(configs["device"])

    tokenizer = AutoTokenizer.from_pretrained(configs["tokenizer"])

    model = ImageCaptionModel(
        embedding_dim=configs["embedding_dim"],
        attention_dim=configs["attention_dim"],
        vocab_size=tokenizer.vocab_size,
        max_seq_len=configs["max_seq_len"],
        num_layers=configs["num_layers"],
        num_heads=configs["num_heads"],
        dropout=configs["dropout"],
    )
    print("Model to {}".format(device))
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optim = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), eps=1e-9)

    # Dataset
    train_dataset = ImageCaptionDataset(
        karpathy_json_path=configs["karpathy_json_path"],
        image_dir=configs["image_dir"],
        tokenizer=tokenizer,
        max_seq_len=configs["max_seq_len"],
        transform=transform, 
        phase="train"
    )
    valid_dataset = ImageCaptionDataset(
        karpathy_json_path=configs["karpathy_json_path"],
        image_dir=configs["image_dir"],
        tokenizer=tokenizer,
        max_seq_len=configs["max_seq_len"],
        transform=transform, 
        phase="val"
    )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=configs["batch_size"],
        shuffle=False
    )

    start_time = time.time()
    # Train
    log = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optim=optim,
        criterion=criterion,
        n_epochs=configs["n_epochs"],
        tokenizer=tokenizer,
        device=device,
        model_path=configs["model_path"],
        log_path=configs["log_path"],
        early_stopping=configs["early_stopping"]
    )

    print(f"======================== Training finished: {timedelta(seconds=int(time.time()-start_time))} ========================")
    print(f"---- Training | Best BLEU-4: {log['best_train_bleu4']:.5f} | Best Loss: {min(log['train_loss']):.5f}")
    print(f"---- Validation | Best BLEU-4: {log['best_val_bleu4']:.5f} | Best Loss: {min(log['val_loss']):.5f}")
    print(f"---- Best epoch: {log['best_epoch']}")

    # Save log
    with open(configs["log_path"], "w") as f:
        json.dump(log, f)
    
    # Visualize loss
    visualize_log(log, configs)


if __name__ == "__main__":
    main()