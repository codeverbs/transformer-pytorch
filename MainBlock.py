import torch.nn as nn
from SelfAttention import SelfAttention

# This class contains the basic transformer block having attention and feed forward network
class GeneralBlock(nn.module):
    def __init__(self, embed_size, attention_heads, dropout, ff_exp):
        super(GeneralBlock, self).__init__()
        self.attention = SelfAttention(embed_size, attention_heads)
        # Initialize layer normalization
        self.layer_norm_01 = nn.LayerNorm(embed_size)
        self.layer_norm_02 = nn.LayerNorm(embed_size)
        # Setting up our Feed Forward Neural Network With Single Hidden Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_exp*embed_size),
            nn.ReLU(),
            nn.Linear(ff_exp*embed_size, embed_size),
        )
        # Initialize the dropout
        # We apply dropout normalization as it helps in preventing overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        # Applying layer normalization to self-attention along with dropout norm
        layer_norm1 = self.layer_norm_01(queries + attention)
        output1 = self.dropout(layer_norm1)
        # Sending normalized data to our feed-forward neural network
        feedforward = self.feed_forward(output1)
        # Applying layer normalization and dropout norm to neural network output
        layer_norm2 = self.layer_norm_02(output1 + feedforward)
        output2 = self.dropout(layer_norm2)
        return output2