import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.multihead_attention import MultiHeadAttention

# Using the implemented MultiHeadSelfAttention
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        """
        Initialize the Feed-Forward network.
        
        Args:
            embed_dim (int): The input and output dimensionality.
            ff_dim (int): The inner dimensionality of the feed-forward network.
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Initialize an encoder layer.
        
        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            ff_dim (int): The inner dimensionality of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class EncoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_length, num_classes, dropout=0.1):
        """
        Initialize the Encoder-Only model.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): The dimensionality of the embeddings.
            num_heads (int): The number of attention heads.
            ff_dim (int): The inner dimensionality of the feed-forward network.
            num_layers (int): The number of encoder layers.
            max_seq_length (int): Maximum sequence length for positional encoding.
            num_classes (int): Number of classification categories.
            dropout (float): Dropout rate.
        """
        super(EncoderModel, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.create_positional_encoding(max_seq_length, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Optional - Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
    def create_positional_encoding(self, max_seq_length, embed_dim):
        """
        Create positional encodings for the input.
        """
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        """
        Perform the forward pass of the encoder model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification head
        x = self.classification_head(x)
        
        return x

# Example usage:
vocab_size = 10000
embed_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 6
max_seq_length = 512
num_classes = 10
batch_size = 32
seq_len = 128

model = EncoderModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_length, num_classes)
input_data = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(input_data)
print(output.shape)  # Expected: torch.Size([32, 10])