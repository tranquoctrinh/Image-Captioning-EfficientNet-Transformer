<!-- # Image Captioning
Image Captioning by deep learning model with Encoder as Efficientnet and Decoder as Decoder of Transformer -->
# Table of Contents
- [Table of Contents](#table-of-contents)
- [1. Objective](#1-objective)
- [2. Model](#2-model)
- [3. Dataset](#3-dataset)
- [4. Training and Validation: Image Captioning](#4-training-and-validation-image-captioning)
  - [4.1. Training](#41-training)
    - [4.1.1. Images](#411-images)
    - [4.1.2. Captions](#412-captions)
    - [4.1.3. Hyperparameters](#413-hyperparameters)
  - [4.2. Validation](#42-validation)
- [5. Evaluation](#5-evaluation)
- [6. Inferece](#6-inferece)
- [7. Conclusion](#7-conclusion)

# 1. Objective

The objective of this project is to build a model that can generate captions for images.

The directory structure of this project is shown below:
```bash
root/
├── coco/
│   ├── annotations/
│   │   ├── captions_train2014.json
│   │   └── captions_train2014.json
│   ├── karpathy/
│   │   └── dataset_coco.json
│   ├── train2014/
│   └── val2014/
│
└── image_captioning/ # this repository
    ├── images/
    ├── pretrained/
    ├── results/
    ├── caption.py
    ├── datasets.py
    ├── evaluation.py
    ├── models.py
    ├── README.md
    ├── train.py
    └── utils.py
```
# 2. Model

I use Encoder as Efficientnet to extract features from image and Decoder as Transformer to generate caption. But I also change the attention mechanism at step attention encoder output. Instead of using the Multi-Head Attention mechanism, I use the Attention mechanism each step to attend image features.
<figure align="center">
  <p align="center"><img src="./images/model_architecture_trans.png" width="600"/>
    <figcaption><b>Model architecture:</b> <i>The architecture of the model Image Captioning with Encoder as Efficientnet and Decoder as Transformer</i></figcaption>
  </p>
</figure>

# 3. Dataset
I'm using the MSCOCO '14 Dataset. You'd need to download the Training (13GB),  Validation (6GB) and Test (6GB) splits from [MSCOCO](http://cocodataset.org/#download) and place them in the `../coco` directory.

I'm also using Andrej Karpathy's split of the MSCOCO '14 dataset. It contains caption annotations for the MSCOCO, Flickr30k, and Flickr8k datasets. You can download it from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). You'd need to unzip it and place it in the `../coco/karpathy` directory.
In Andrej's split, the images are divided into train, val and test sets with the number of images in each set as shown in the table below:

| Image/Caption | train | val | test |
| :--- | :--- | :--- | :--- |
| Image | 113287 | 5000 | 5000 |
| Caption | 566747 | 25010 | 25010 |



# 4. Training and Validation: Image Captioning
## Pre-processing
### 4.1.1. Images
I preprocessed the images with the following steps:
- Resize the images to 256x256 pixels.
- Convert the images to RGB.
- Normalize the images with mean and standard deviation.
I normalized the image by the mean and standard deviation of the ImageNet images' RGB channels.
```python
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```


### 4.1.2. Captions
Captions are both the target and the inputs of the Decoder as each word is used to generate the next word.

I use BERTTokenizer to tokenize the captions.
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
token = tokenizer(caption, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"][0]
```

For more details, see `datasets.py` file.

## 4.2. Training
### 4.2.1 Model configs

- embedding_dim: 512
- vocab_size: 30522
- max_seq_len: 128
- encoder_layers: 6
- decoder_layers: 12
- num_heads: 8
- dropout: 0.1

### 4.2.2. Hyperparameters

- n_epochs: 25
- batch_size: 24
- learning_rate: 1e-4
- optimizer: Adam
- adam parameters: betas=(0.9, 0.999), eps=1e-9
- loss: CrossEntropyLoss
- metric: bleu-4
- early_stopping: 5

## 4.2. Validation
I evaluate the model on the validation set after each epoch. For each image, I generate a caption and evaluate the BLEU-4 score with list of reference captions by sentence_bleu. And for all the images, I calculate the BLEU-4 score with the corpus_bleu function from NLTK.

You can see the detaile in the `train.py` file. Run `train.py` to train the model.
```bash
python train.py \
    --embedding_dim 512 \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --encoder_layers 6 \
    --decoder_layers 12 \
    --num_heads 8 \
    --dropout 0.1 \
    --model_path ./pretrained/model_image_captioning_eff_transfomer.pt \
    --device cuda:0 \
    --batch_size 24 \
    --n_epochs 25 \
    --learning_rate 1e-4 \
    --early_stopping 5 \
    --image_dir ../coco/ \
    --karpathy_json_path ../coco/karpathy/dataset_coco.json \
    --val_annotation_path ../coco/annotations/captions_val2014.json \
    --log_path ./images/log_training.json \
    --log_visualize_dir ./images/
```

# 5. Evaluation
See the `evaluation.py` file. Run `evaluation.py` to evaluate the model.

```bash
python evaluation.py \
    --embedding_dim 512 \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --encoder_layers 6 \
    --decoder_layers 12 \
    --num_heads 8 \
    --dropout 0.1 \
    --model_path ./pretrained/model_image_captioning_eff_transfomer.pt \
    --device cuda:0 \
    --image_dir ../coco/ \
    --karpathy_json_path ../coco/karpathy/dataset_coco.json \
    --val_annotation_path ../coco/annotations/captions_val2014.json \
    --output_dir ./results/
```

To evaluate the model, I used the [pycocoevalcap package](https://github.com/salaniz/pycocoevalcap). Install it by `pip install pycocoevalcap`. And this package need to be Java 1.8.0 installed.

```bash
sudo apt-get update
sudo apt-get install openjdk-8-jdk
java -version

pip install pycocoevalcap
```

I use beam search to generate captions with beam size of 3. I use the BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L, CIDEr, and SPICE score to evaluate the model. The results on the test set (5000 images) are shown below.

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.675 | 0.504 | 0.372 | 0.273 | 0.259 | 0.521 | 0.933 | 0.190 |


# 6. Inferece
See the file `caption.py`. Run `caption.py` to generate captions for the test images. If you don't have resouces for training, you can download the pretrained model from [here](https://drive.google.com/file/d/1CcCPJ-cCfosGe7iOaPPbA3EJzNxsKuc_/view?usp=sharing).

```bash
python caption.py \
    --embedding_dim 512 \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --encoder_layers 6 \
    --decoder_layers 12 \
    --num_heads 8 \
    --dropout 0.1 \
    --model_path ./pretrained/model_image_captioning_eff_transfomer.pt \
    --device cuda:0 \
    --beam_size 3 
```

```python
from evaluation import generate_caption

cap = generate_caption(
    model=model,
    image_path=image_path,
    transform=transform,
    tokenizer=tokenizer,
    max_seq_len=args.max_seq_len,
    beam_size=args.beam_size,
    device=device
)
print("--- Caption: {}".format(cap))
```

**Some examples of captions generated from COCO images are shown below.**
<table>
  <tr>
    <td><img src="images/test_1.jpg" ></td>
    <td><img src="images/test_2.jpg" ></td>
    <td><img src="images/test_5.jpg" ></td>
  </tr>
  <tr>
    <td>A bride and groom cutting a wedding cake. </td>
     <td>A man riding a wave on top of a surfboard.</td>
     <td>A close up view of a keyboard and a mouse.</td>
  </tr>
 </table>
 
<table>
  <tr>
    <td><img src="images/test_3.jpg" ></td>
    <td><img src="images/test_4.jpg" ></td>
    <td><img src="images/test_7.jpg" ></td>
  </tr>
  <tr>
    <td>A red fire hydrant sitting on the side of a street.</td>
     <td>A woman holding a hot dog in front of her.</td>
     <td>A red stop sign sitting on top of a metal pole.</td>
  </tr>
 </table>

 <table>
  <tr>
    <td><img src="images/test_6.jpg" ></td>
    <td><img src="images/test_11.jpg" ></td>
    <td><img src="images/test_9.jpg" ></td>
  </tr>
  <tr>
    <td>A little girl sitting in front of a laptop computer on a desk. </td>
     <td>A group of people playing a game of frisbee in a park.</td>
     <td>A group of people standing on top of a beach with surfboards.</td>
  </tr>
 </table>

**Some examples of captions generated from other images that are not in the COCO dataset are shown below.**
  <table>
  <tr>
    <td><img src="images/pogba.jpg" ></td>
    <td><img src="images/baby.jpg" ></td>
    <td><img src="images/test.jpg" ></td>
  </tr>
  <tr>
    <td>A soccer player kicking a soccer ball on a green field. </td>
     <td>A baby sleeping in a bed with a blanket on its head.</td>
     <td>A brown and white dog standing on top of a grass-covered field.</td>
  </tr>
 </table>

 
# 7. Conclusion