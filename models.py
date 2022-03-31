import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# import efficientnet model
from efficientnet_pytorch import EfficientNet
import math
from torch.autograd import Variable
import numpy as np

# Embedding the input sequence
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Norm layer
class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)

# The positional encoding vector
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/embedding_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x*math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        # Add the positional encoding vector to the embedding vector
        x = x + pe
        x = self.dropout(x)
        return x


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

# SoftAttention
class SoftAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(SoftAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, memory, decoder_hidden):
        batch_size = memory.size(0) # batch size
        memory = memory.view(batch_size, -1, self.encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        att1 = self.encoder_att(memory)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, max_seq_len, attention_dim)
        att = self.softmax(torch.matmul(att1, att2.transpose(1, 2)))/np.sqrt(self.encoder_dim)
        # (batch_size, num_pixels, max_seq_len)
        return att

# Transformer decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, attention_dim, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = SoftAttention(embedding_dim, embedding_dim, attention_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim + 49, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim + 49)

    def forward(self, x, memory, target_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attention(x2, x2, x2, target_mask))
        x2 = self.norm1(x)
        att = self.dropout2(self.encoder_attention(memory, x2)) # (batch_size, num_pixels, max_seq_len)
        att = att.transpose(1, 2) # (batch_size, max_seq_len, num_pixels)
        x2 = torch.cat([x, att], dim=-1) # (batch_size, max_seq_len, embedding_dim + num_pixels)
        x2 = self.norm3(x2)
        x = x + self.dropout3(self.feed_forward(x2))
        return x


# Decoder model image captioning with Attention
class Decoder(nn.Module):
    def __init__(self, embedding_dim, attention_dim, vocab_size, max_seq_len, num_layers, num_heads, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, attention_dim, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)

    def forward(self, target, memory, target_mask):
        # target: (batch_size, seq_len)
        # memory is the encoder output
        x = self.embed(target)
        # Add the position embeddings
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, memory, target_mask)
        x = self.norm(x)
        return x

# Model image captioning with Attention
class ImageCaptioning(nn.Module):
    def __init__(self, embedding_dim=512, attention_dim=256, vocab_size=0, max_seq_len=256, num_layers=8, num_heads=8, dropout=0.1):
        super(ImageCaptioning, self).__init__()
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
        self.encoder = Encoder()
        self.decoder = Decoder(embedding_dim, attention_dim, vocab_size, max_seq_len, num_layers, num_heads, dropout)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.norm = Norm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, image, captions):
        # image: (batch_size, embedding_dim)
        # captions: (batch_size, max_len)
        # lengths: (batch_size)
        batch_size = image.size(0)
        # Encode the image
        encoder_output = self.encoder(image)
        # Create captions mask
        target_mask = self.make_mask(captions)
        # Decode the image
        decoder_output = self.decoder(captions, encoder_output, target_mask)
        # Apply the linear projection
        decoder_output = self.fc(decoder_output)
        return decoder_output
    
    def make_mask(self, target_ids):
        batch_size, len_target = target_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=target_ids.device), diagonal=1)).bool()
        return subsequent_mask

def main():
    # transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = 4
    max_seq_len = 128
    vocab_size = 123456
    model = ImageCaptioning(embedding_dim=2048, attention_dim=256, vocab_size=vocab_size, num_layers=8, num_heads=8, dropout=0.1)
    out = model(torch.randn(batch_size, 3, 224, 224), torch.randint(0, vocab_size, (batch_size, max_seq_len)))

    import ipdb; ipdb.set_trace()
if __name__ == "__main__":
    main()