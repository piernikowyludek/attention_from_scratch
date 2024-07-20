import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # Initialize the layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)

        # Apply linear transformation to get queries, keys, and values
        queries = self.query_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        keys = self.key_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        values = self.value_layer(x)  # shape: (batch_size, seq_length, embed_dim)

        # Calculate attention scores
        # scores shape: (batch_size, seq_length, seq_length)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale

        # Apply softmax to get attention weights
        # attention_weights shape: (batch_size, seq_length, seq_length)
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the attended values
        # attended_values shape: (batch_size, seq_length, embed_dim)
        attended_values = torch.matmul(attention_weights, values)

        return attended_values
    

if __name__ == "__main__":
    batch_size = 2
    seq_length = 3
    embed_dim = 4

    # Create a random tensor with shape (batch_size, seq_length, embed_dim)
    x = torch.rand(batch_size, seq_length, embed_dim)

    # Initialize the self-attention layer
    self_attention = SelfAttention(embed_dim)

    # Forward pass
    output = self_attention(x)

    print("Input:\n", x.size())
    print("Output:\n", output.size())
