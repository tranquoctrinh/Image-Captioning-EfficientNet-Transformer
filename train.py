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
from torchvision import transforms
import json
from datasets import ImageCaptionDataset
from models import ImageCaptionModel


def train_epoch(model, train_loader, optim, criterion, epoch, device):
    model.train()
    total_loss = []
    bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch+1}")
    for i, batch in bar:
        image, caption = batch["image"].to(device), batch["caption"].to(device)
        target_input = caption[:, :-1]
        target_mask = model.make_mask(target_input)
        preds = model(image, target_input)
        optim.zero_grad()
        gold = caption[:, 1:].contiguous().view(-1)
        loss = criterion(preds.view(-1, preds.size(-1)), gold)
        loss.backward()
        optim.step()
        total_loss.append(loss.item())
        bar.set_postfix(loss=total_loss[-1])
    
    return sum(total_loss) / len(total_loss)
    

def validate_epoch(model, valid_loader, tokenizer, epoch, device):
    model.eval()
    hypotheses = []
    references = []
    batch_bleu4 = []
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validating epoch {epoch+1}")
    for i, batch in bar:
        image, caption, all_caps = batch["image"].to(device), batch["caption"].to(device), batch["all_captions_seq"]
        target_input = caption[:, :-1]
        target_mask = model.make_mask(target_input)
        preds = model(image, target_input)
        preds = F.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1)
        preds = preds.detach().cpu().numpy()
        caps = [tokenizer.decode(cap, skip_special_tokens=True) for cap in preds]
        hypo = [cap.split() for cap in caps]
        
        batch_size = len(hypo)
        ref = []
        for i in range(batch_size):
            ri = []
            for j in range(len(all_caps)):
                if all_caps[j][i]:
                    ri.append(all_caps[j][i].split())
            ref.append(ri)
        
        from nltk.translate.bleu_score import SmoothingFunction
        smoothie = SmoothingFunction()

        batch_bleu4.append(corpus_bleu(ref, hypo, smoothing_function=smoothie.method4))
        hypotheses += hypo
        references += ref
        bar.set_postfix(bleu4=sum(batch_bleu4) / len(batch_bleu4))
    
    return corpus_bleu(references, hypotheses, smoothing_function=smoothie.method4)

def train(model, train_loader, valid_loader, optim, criterion, n_epochs, tokenizer, device, model_path):
    model.train()
    lst_train_loss, lst_bleu4 = [], []
    best_bleu4 = -np.Inf
    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optim=optim,
            criterion=criterion,
            epoch=epoch,
            device=device
        )
        current_bleu4 = validate_epoch(
            model=model,
            valid_loader=valid_loader,
            tokenizer=tokenizer,
            epoch=epoch,
            device=device
        )
        print(f"---- Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.5f} | Validation BLEU-4: {current_bleu4:.5f} | Best BLEU-4: {best_bleu4:.5f}")
        lst_train_loss.append(train_loss)
        lst_bleu4.append(current_bleu4)
        if current_bleu4 > best_bleu4:
            best_bleu4 = current_bleu4
            # Save Model with best validation bleu4
            torch.save(model.state_dict(), model_path)
            print("-------- Detect improment and save the best model --------")
        
        torch.cuda.empty_cache()
    
    return lst_train_loss, lst_bleu4
        

def main():
    # configs
    from utils import configs
    
    device = torch.device(configs["device"])

    tokenizer = AutoTokenizer.from_pretrained(configs["tokenizer"])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

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

    # Train
    train_loss, bleu4 = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optim=optim,
        criterion=criterion,
        n_epochs=configs["n_epochs"],
        tokenizer=tokenizer,
        device=device,
        model_path=configs["model_path"]
    )
    print(f"---- Training Loss: {train_loss}")
    print(f"---- Validation BLEU-4: {bleu4}")

    log = {
        "train_loss": train_loss,
        "bleu4": bleu4
    }
    json.dump(log, open("./log_image_captioning_eff_transformer.json", "w"))
    # visualize line plot train loss and validation bleu4
    plt.plot(train_loss)
    plt.title("Train Loss")
    plt.savefig("./images/train_loss_image_captioning_eff_transformer.png")
    plt.clf()
    plt.plot(bleu4)
    plt.title("Validation BLEU-4")
    plt.savefig("./images/valid_bleu4_image_captioning_eff_transformer.png")
    

if __name__ == "__main__":
    main()