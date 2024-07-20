import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Initialize the layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Linear layer to combine the outputs of multiple heads
        self.out_layer = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)

        batch_size = x.size(0)

        # Apply linear transformation to get queries, keys, and values
        queries = self.query_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        keys = self.key_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        values = self.value_layer(x)  # shape: (batch_size, seq_length, embed_dim)

        # Split into multiple heads
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Now, queries, keys, and values have shape: (batch_size, num_heads, seq_length, head_dim)

        # Calculate attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        # scores shape: (batch_size, num_heads, seq_length, seq_length)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights shape: (batch_size, num_heads, seq_length, seq_length)

        # Compute the attended values
        attended_values = torch.matmul(attention_weights, values)
        # attended_values shape: (batch_size, num_heads, seq_length, head_dim)

        # Concatenate the multiple heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # attended_values shape: (batch_size, seq_length, embed_dim)

        # Apply the final linear layer
        output = self.out_layer(attended_values)
        # output shape: (batch_size, seq_length, embed_dim)

        return output
    

class PerformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size=256):
        super(PerformerAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_size = kernel_size

        # Initialize the layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Linear layer to combine the outputs of multiple heads
        self.out_layer = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        # Fixed random matrix for feature map approximation
        self.random_matrix = torch.randn((self.head_dim, self.kernel_size))

    def _feature_map(self, x):
        # Use the fixed random matrix for both queries and keys
        return F.relu(torch.matmul(x, self.random_matrix.to(x.device)))

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)

        batch_size = x.size(0)

        # Apply linear transformation to get queries, keys, and values
        queries = self.query_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        keys = self.key_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        values = self.value_layer(x)  # shape: (batch_size, seq_length, embed_dim)

        # Split into multiple heads
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Now, queries, keys, and values have shape: (batch_size, num_heads, seq_length, head_dim)

        # Apply the feature map to approximate the softmax kernel
        queries = self._feature_map(queries)
        keys = self._feature_map(keys)

        # Calculate attention scores using the feature maps
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        # scores shape: (batch_size, num_heads, seq_length, seq_length)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights shape: (batch_size, num_heads, seq_length, seq_length)

        # Compute the attended values
        attended_values = torch.matmul(attention_weights, values)
        # attended_values shape: (batch_size, num_heads, seq_length, head_dim)

        # Concatenate the multiple heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # attended_values shape: (batch_size, seq_length, embed_dim)

        # Apply the final linear layer
        output = self.out_layer(attended_values)
        # output shape: (batch_size, seq_length, embed_dim)

        return output


class LinformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, proj_dim):
        super(LinformerAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj_dim = proj_dim

        # Initialize the layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Linear layer to project keys and values
        self.key_projection = nn.Linear(embed_dim, proj_dim)
        self.value_projection = nn.Linear(embed_dim, proj_dim)

        # Linear layer to combine the outputs of multiple heads
        self.out_layer = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)

        batch_size = x.size(0)
        seq_length = x.size(1)

        # Apply linear transformation to get queries, keys, and values
        queries = self.query_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        keys = self.key_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        values = self.value_layer(x)  # shape: (batch_size, seq_length, embed_dim)

        # Project keys and values
        keys = self.key_projection(keys)  # shape: (batch_size, proj_dim, embed_dim)
        values = self.value_projection(values)  # shape: (batch_size, proj_dim, embed_dim)

        # Split into multiple heads
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Now, queries, keys, and values have shape: (batch_size, num_heads, seq_length, head_dim)

        # Calculate attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        # scores shape: (batch_size, num_heads, seq_length, proj_dim)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights shape: (batch_size, num_heads, seq_length, proj_dim)

        # Compute the attended values
        attended_values = torch.matmul(attention_weights, values)
        # attended_values shape: (batch_size, num_heads, seq_length, head_dim)

        # Concatenate the multiple heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # attended_values shape: (batch_size, seq_length, embed_dim)

        # Apply the final linear layer
        output = self.out_layer(attended_values)
        # output shape: (batch_size, seq_length, embed_dim)

        return output
    

class LocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super(LocalAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        # Initialize the layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Linear layer to combine the outputs of multiple heads
        self.out_layer = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)

        batch_size, seq_length, embed_dim = x.size()

        # Apply linear transformation to get queries, keys, and values
        queries = self.query_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        keys = self.key_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        values = self.value_layer(x)  # shape: (batch_size, seq_length, embed_dim)

        # Split into multiple heads
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Now, queries, keys, and values have shape: (batch_size, num_heads, seq_length, head_dim)

        # Initialize the output tensor
        output = torch.zeros_like(values)

        # Calculate local attention scores and apply them
        for i in range(seq_length):
            start = max(0, i - self.window_size)
            end = min(seq_length, i + self.window_size + 1)

            # Extract the relevant keys and values
            keys_slice = keys[:, :, start:end, :]
            values_slice = values[:, :, start:end, :]

            # Calculate attention scores
            scores = torch.matmul(queries[:, :, i, :].unsqueeze(2), keys_slice.transpose(-2, -1)) / self.scale
            # scores shape: (batch_size, num_heads, 1, window_size*2+1)

            # Apply softmax to get attention weights
            attention_weights = F.softmax(scores, dim=-1)
            # attention_weights shape: (batch_size, num_heads, 1, window_size*2+1)

            # Compute the attended values
            attended_values = torch.matmul(attention_weights, values_slice)
            # attended_values shape: (batch_size, num_heads, 1, head_dim)

            # Store the result in the output tensor
            output[:, :, i, :] = attended_values.squeeze(2)

        # Concatenate the multiple heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        # output shape: (batch_size, seq_length, embed_dim)

        # Apply the final linear layer
        output = self.out_layer(output)
        # output shape: (batch_size, seq_length, embed_dim)

        return output
    

class DynamicSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, top_k):
        super(DynamicSparseAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k

        # Initialize the layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Linear layer to combine the outputs of multiple heads
        self.out_layer = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for attention scores
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)

        batch_size, seq_length, embed_dim = x.size()

        # Apply linear transformation to get queries, keys, and values
        queries = self.query_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        keys = self.key_layer(x)  # shape: (batch_size, seq_length, embed_dim)
        values = self.value_layer(x)  # shape: (batch_size, seq_length, embed_dim)

        # Split into multiple heads
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Now, queries, keys, and values have shape: (batch_size, num_heads, seq_length, head_dim)

        # Initialize the output tensor
        output = torch.zeros_like(values)

        # Calculate attention scores and apply top-k sparse attention
        for i in range(seq_length):
            # Calculate attention scores for the current query
            scores = torch.matmul(queries[:, :, i, :].unsqueeze(2), keys.transpose(-2, -1)) / self.scale
            # scores shape: (batch_size, num_heads, 1, seq_length)

            # Find the top-k indices based on attention scores
            top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)

            # Gather the top-k keys and values
            top_k_keys = torch.gather(keys, 2, top_k_indices.expand(-1, -1, -1, self.head_dim))
            top_k_values = torch.gather(values, 2, top_k_indices.expand(-1, -1, -1, self.head_dim))

            # Recalculate attention scores for the top-k keys
            scores = torch.matmul(queries[:, :, i, :].unsqueeze(2), top_k_keys.transpose(-2, -1)) / self.scale
            attention_weights = F.softmax(scores, dim=-1)
            # attention_weights shape: (batch_size, num_heads, 1, top_k)

            # Compute the attended values
            attended_values = torch.matmul(attention_weights, top_k_values)
            # attended_values shape: (batch_size, num_heads, 1, head_dim)

            # Store the result in the output tensor
            output[:, :, i, :] = attended_values.squeeze(2)

        # Concatenate the multiple heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        # output shape: (batch_size, seq_length, embed_dim)

        # Apply the final linear layer
        output = self.out_layer(output)
        # output shape: (batch_size, seq_length, embed_dim)

        return output


if __name__ == "__main__":
    batch_size = 2
    seq_length = 3
    embed_dim = 8
    num_heads = 2

    # Create a random tensor with shape (batch_size, seq_length, embed_dim)
    x = torch.rand(batch_size, seq_length, embed_dim)

    # Initialize the multi-head attention layer
    multi_head_attention = MultiHeadAttention(embed_dim, num_heads)

    # Forward pass
    output = multi_head_attention(x)

    print("Input:\n", x.size())
    print("Output:\n", output.size())