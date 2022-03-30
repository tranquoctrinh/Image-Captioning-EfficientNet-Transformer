import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# import efficientnet model
from efficientnet_pytorch import EfficientNet

# Encoder model image captioning with EfficientNet
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')

    def forward(self, images):
        # images: (batch_size, 3, 224, 224)
        features = self.model.extract_features(images)
        # features: (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)
        # features: (batch_size, 7, 7, 2048)
        return features

# SoftAttention
class SoftAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, vocab_size):
        super(SoftAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.linear = nn.Linear(encoder_dim, decoder_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features, captions):
        # features: (batch_size, 7, 7, 2048)
        # captions: (batch_size, max_len, vocab_size)
        batch_size = features.size(0)
        max_len = captions.size(1)
        # (batch_size, max_len, decoder_dim)
        embeddings = self.linear(features.view(batch_size, -1))
        # (batch_size, max_len, decoder_dim)
        embeddings = embeddings.view(batch_size, max_len, self.decoder_dim)
        # (batch_size, max_len, decoder_dim)
        attention = torch.bmm(captions, embeddings)
        # (batch_size, max_len, 7, 7)
        attention = F.softmax(attention, dim=1)
        # (batch_size, max_len, 7, 7)
        context = torch.bmm(attention, features)
        # (batch_size, max_len, 7, 7)
        return context
        

# Self-attention layer
class SelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, value)

        return output

# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output

# Transformer decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = SoftAttention(embedding_dim, hidden_dim, attention_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim)

    def forward(self, x, memory, source_mask, target_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attention(x2, x2, x2, target_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.encoder_attention(x2, memory, memory, source_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))
        return x

# Decoder model image captioning with Attention
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, attention_dim, vocab_size, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.attention = SoftAttention(embedding_dim, hidden_dim, attention_dim)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)
        self.norm = Norm(embedding_dim)

    def forward(self, target, memory, source_mask, target_mask):
        # target: (batch_size, seq_len)
        # memory is the encoder output
        x = self.embed(target)
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Model image captioning with Attention
class ImageCaptioning(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, attention_dim, vocab_size, num_layers, num_heads, dropout=0.1):
        super(ImageCaptioning, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, attention_dim, vocab_size, num_layers, dropout)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.norm = Norm(embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, image, captions, lengths):
        # image: (batch_size, embedding_dim)
        # captions: (batch_size, max_len)
        # lengths: (batch_size)
        batch_size = image.size(0)
        # Encode the image
        encoder_output = self.encoder(image)
        # Create the mask
        target_mask = torch.zeros(batch_size, captions.size(1), dtype=torch.float)
        for i, length in enumerate(lengths):
            target_mask[i, length:] = 1
        # Decode the image
        decoder_output = self.decoder(captions, encoder_output, target_mask, target_mask)
        # Apply the linear projection
        decoder_output = self.fc(decoder_output)
        return decoder_output



def main():
    # transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])