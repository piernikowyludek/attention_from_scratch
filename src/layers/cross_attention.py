import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim

        # Initialize the layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

    def forward(self, query, context):
        # query shape: (batch_size, query_length, embed_dim)
        # context shape: (batch_size, context_length, embed_dim)

        # Apply linear transformation to get queries, keys, and values
        queries = self.query_layer(query)  # shape: (batch_size, query_length, embed_dim)
        keys = self.key_layer(context)  # shape: (batch_size, context_length, embed_dim)
        values = self.value_layer(context)  # shape: (batch_size, context_length, embed_dim)

        # Calculate attention scores
        # scores shape: (batch_size, query_length, context_length)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale

        # Apply softmax to get attention weights
        # attention_weights shape: (batch_size, query_length, context_length)
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the attended values
        # attended_values shape: (batch_size, query_length, embed_dim)
        attended_values = torch.matmul(attention_weights, values)

        return attended_values


if __name__ == "__main__":
    batch_size = 2
    query_length = 3
    context_length = 4
    embed_dim = 5

    # Create random tensors for query and context
    query = torch.rand(batch_size, query_length, embed_dim)
    context = torch.rand(batch_size, context_length, embed_dim)

    # Initialize the cross-attention layer
    cross_attention = CrossAttention(embed_dim)

    # Forward pass
    output = cross_attention(query, context)

    print("Query:\n", query.size())
    print("Context:\n", context.size())
    print("Output:\n", output.size())