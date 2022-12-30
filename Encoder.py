import torch
import torch.nn as nn
from GeneralBlock import GeneralBlock

# This is the class which holds the code for Transformer's Encoder Module
class Encoder(nn.Module):
    def __init__(self, embed_size, transformer_layers, attention_heads, vocab_size, max_length, ff_exp, dropout, device):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device # Device to use; CPU or CUDA
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size) # Max length specifies the max size the input sequence can have
        # Initializing encoder blocks for the specified number of times
        self.encoder_layers = nn.ModuleList(
            [
                GeneralBlock(embed_size, attention_heads, ff_exp, dropout)
                for _ in range(transformer_layers)
            ]
        )
        # Initializing dropout for output normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        sample_count, input_len = x.shape
        # Creating positioning array based on total samples count and input sequence length
        # eg. for 2 training examples of total length 5 each the positions mat becomes
        # positions = [[0,1,2,3,4], [0,1,2,3,4]]
        # The above array assigns unique positions to every word in input training sample
        positions = torch.arange(0, input_len).expand(sample_count, input_len).to(self.device)
        # We use embeddings layer to assign weights to every input word per training sample
        # We have seperate embedding for words and for their position in the setence, each
        # embedding has its own weight matrix having size of 'embed_size'
        input_embeddings = self.word_embedding(x) + self.position_embedding(positions)
        encoder_output = self.dropout(input_embeddings)
        # We pass our Keys, Queries & Values to Encoder_Layer
        # Note that the K,Q,V pair remains same for Encoder
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, encoder_output, encoder_output, mask)
        # Finally, return the output
        return encoder_output