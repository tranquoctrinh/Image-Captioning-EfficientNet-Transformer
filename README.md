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
- [6. Conclusion](#6-conclusion)

# 1. Objective
The objective of this project is to build a model that can generate captions for images.

# 2. Model
I use Encoder as Efficientnet to extract features from image and Decoder as Transformer to generate caption. But I also change the attention mechanism at step attention encoder output. Instead of using the Multi-Head Attention mechanism, I use the Attention mechanism each step to attend image features.
<figure align="center">
    <img src="./images/model_architecture.png" width="600"/>
    <figcaption>Model architecture: The architecture of the model Image Captioning with Encoder as Efficientnet and Decoder as Transformer</figcaption>
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
## 4.1. Training
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

token = tokenizer(caption, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"][0]
```

For more details, see `datasets.py`.

### 4.1.3. Hyperparameters
- Epochs: 50
- Batch size: 32
- Max sequence length: 128
- Learning rate: 1e-4
- Optimizer: Adam
- Adam weight decay: 0.01
- Adam beta: (0.9, 0.98)
- Adam eps: 1e-09
- Loss: Cross-entropy
- Metric: BLEU-4
- Early stopping: 5

All the hyperparameters are defined in `utils.py`. So, you can change the hyperparameters by yourself easily.

## 4.2. Validation
I evaluate the model on the validation set after each epoch. For each image, I generate a caption and evaluate the BLEU-4 score with list of reference captions by sentence_bleu. And for all the images, I calculate the BLEU-4 score with the corpus_bleu function from NLTK.

You can see the detaile in the `train.py` file.

# 5. Evaluation
To evaluate the model, I used the [pycocoevalcap package](https://github.com/salaniz/pycocoevalcap).

I use beam search to generate captions with beam size of 3, 4, 5. I use the BLEU-4, METEOR, ROUGE-L, CIDEr, and SPICE score to evaluate the model. The results on the test set (5000 images) are shown below.

| Metrics | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Beam Size 3 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| Beam Size 4 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| Beam Size 5 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |

# 6. Conclusion