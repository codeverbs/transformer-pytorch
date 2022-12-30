import torch
import torch.nn as nn
from GeneralBlock import GeneralBlock
from SelfAttention import SelfAttention

# We create a new class encapsulating our GeneralBlock class for the implementation of Decoder
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, attention_heads, ff_exp, dropout):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, attention_heads)
        self.general_block = GeneralBlock(embed_size, attention_heads, dropout, ff_exp)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # Sending A Masked Input to Attention
        attention = self.attention(x, x, x, trg_mask)
        # Performing Layer Norm on the Query ouput from Decoder Block
        query = self.dropout(self.norm(attention + x))
        # Now using the general transformer block which uses queries from Decoder and
        # Key, Values from Encoder
        output = self.general_block(value, key, query, src_mask)
        return output


# This class contains all the code required to run a transformer's decoder which uses encoder's output
class Decoder(nn.Module):
    def __init__(self, embed_size, tranformer_layers, attention_heads, vocab_size, max_length, ff_exp, dropout, device):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size) # Max length specifies the max size the ouput sequence can have
        # Initializing decoder blocks for the specified number of times
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, attention_heads, ff_exp, dropout)
                for _ in range(tranformer_layers)
            ]
        )
        self.output_layer = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        sample_count, input_len = x.shape
        # Creating positioning array based on total samples count and input sequence length
        # eg. for 2 training examples of total length 5 each the positions mat becomes
        # positions = [[0,1,2,3,4], [0,1,2,3,4]]
        # The above array assigns unique positions to every word in input training sample
        positions = torch.arange(0, input_len).expand(sample_count, input_len).to(self.device)
        # We use embeddings layer to assign weights to every input word per training sample
        # We have seperate embedding for words and for their position in the setence, each
        # embedding has its own weight matrix having size of 'embed_size'
        output_embeddings = self.word_embedding(x) + self.position_embedding(positions) 
        decoder_output = self.dropout(output_embeddings)
        # We pass our Keys, Queries & Values to Decoder_Layer
        # Note that the K,V pair comes from Encoder whereas Q comes from Decoder
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, encoder_output, src_mask, trg_mask)
        # Pass the final output from decoder through a linear layer
        decoder_output = self.output_layer(decoder_output)
        return decoder_output