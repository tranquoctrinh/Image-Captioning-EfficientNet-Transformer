import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms


configs = {
    "batch_size": 32,
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

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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


from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def metric_scores(annotation_path, prediction_path):
    # annotation_file = 'captions_val2014.json'
    # results_file = 'captions_val2014_fakecap_results.json'
    # format result file
    # {'image_id': 1, 'caption': 'a caption'}

    results = {}
    # create coco object and coco_result object
    coco = COCO(annotation_path)
    coco_result = coco.loadRes(prediction_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        # print(f'{metric}: {score:.3f}')
        dct[metric] = score
    
    return dct