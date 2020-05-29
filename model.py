import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def padding_mask(inputs, padding_idx=0):
    mask = (inputs == padding_idx).bool()
    return mask


def get_decoder_padding_mask(inputs, padding_idx=0):
    inputs = inputs[:, 0, :]
    mask = padding_mask(inputs, padding_idx)
    return mask


def square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size))).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class ConvBNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, activation=None):
        super(ConvBNBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        # when kernel_size = 5, padding = 2
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_dim,  momentum=0.1, eps=1e-5)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.activation == 'tanh':
            x = self.tanh(x)

        x = self.dropout(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1, activation='relu'):
        super(FeedForward, self).__init__()
        self.linear1 = Linear(d_model, d_ff, w_init=activation)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))
        output = self.dropout(output)

        # residual connection
        output = x + output

        output = self.norm(output)

        return output


class EncoderLayer(nn.Module):
    """Transformer EncoderLayer"""

    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, attn_mask=None, padding_mask=None):
        output, attention = self.attn(x, x, x, attn_mask=attn_mask,
                                      key_padding_mask=padding_mask)
        output = self.ff(output)

        return output, attention


class DecoderLayer(nn.Module):
    """Transformer DecoderLayer"""

    def __init__(self, d_model, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_outputs, dec_attn_mask=None, enc_padding_mask=None, dec_padding_mask=None, ):

        output, self_attention = self.attn(
            x, x, x, key_padding_mask=dec_padding_mask, attn_mask=dec_attn_mask)
        output, context_attention = self.attn(
            output, enc_outputs, enc_outputs, key_padding_mask=enc_padding_mask)

        output = self.ff(output)

        return output, self_attention, context_attention


class Encoder(nn.Module):
    def __init__(self, n_vocab, d_model, embedding_dim, n_layers, n_head, d_ff, dropout, ):
        super(Encoder, self).__init__()
        self.Embedding = nn.Embedding(n_vocab, embedding_dim)
        self.alpha = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList(EncoderLayer(
            d_model, n_head, d_ff, dropout) for _ in range(n_layers))
        self.position_embedding = PositionalEncoding(embedding_dim)

    def forward(self, inputs):
        embedding = self.Embedding(inputs)
        # print(self.position_embedding(embedding).ne(0).shape, 'emmmmm')
        embedding = (embedding + self.alpha *
                     self.position_embedding(embedding)).transpose(0, 1)

        mask = padding_mask(inputs)
        print(mask.shape, 'emmmm')

        attentions = []
        output = embedding
        for layer in self.layers:
            output, attention = layer(output, padding_mask=mask)
            attentions.append(attention)
        return output, attentions, mask
        # return inputs


class Prenet(nn.Module):
    """Some Information about Prenet"""

    def __init__(self, n_mel_channels, d_model):
        super(Prenet, self).__init__()
        self.linear1 = Linear(n_mel_channels, d_model, bias=False)
        self.linear2 = Linear(d_model, d_model, bias=False)

    def forward(self, x):
        x = F.dropout(F.relu(self.linear1(x)), p=0.5, training=True)
        x = F.dropout(F.relu(self.linear2(x)), p=0.5, training=True)

        return x


class Decoder(nn.Module):
    def __init__(self, n_vocab, d_model, n_mel_channels, n_layers, n_head, d_ff, dropout,):
        super(Decoder, self).__init__()
        self.prenet = Prenet(n_mel_channels, d_model)
        self.position_embedding = PositionalEncoding(d_model)

        self.alpha = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList(DecoderLayer(
            d_model, n_head, d_ff, dropout) for _ in range(n_layers))

    def forward(self, mels, enc_outputs, text_mask):
        mel_inputs = F.pad(mels, (1, -1)).transpose(1, 2)
        inputs = self.prenet(mel_inputs)
        inputs = (inputs + self.alpha *
                  self.position_embedding(inputs)).transpose(0, 1)

        padding_mask = get_decoder_padding_mask(mels)
        # diag_mask = torch.triu(mels.new_ones(T, T)).transpose(0, 1)
        # diag_mask[diag_mask == 0] = -float('inf')
        # diag_mask[diag_mask == 1] = 0
        diag_mask = square_subsequent_mask(mels.size(2))

        output = inputs
        attentions, context_attentions = [], []
        for layer in self.layers:
            dec_output, attention, context_attention = layer(
                output, enc_outputs, dec_attn_mask=diag_mask, enc_padding_mask=text_mask, dec_padding_mask=padding_mask)
            attentions.append(attention)
            context_attentions.append(context_attention)

        return output, attentions, context_attentions


class Postnet(nn.Module):
    def __init__(self, postnet_dim, n_mel_channels, n_convs=5):
        super(Postnet, self).__init__()
        # self.convs = nn.ModuleList()
        self.conv_list = []
        self.conv_list.append(
            ConvBNBlock(n_mel_channels, postnet_dim, kernel_size=5, activation='tanh'))
        for _ in range(1, n_convs-1):
            self.conv_list.append(ConvBNBlock(
                postnet_dim, postnet_dim, kernel_size=5, activation='tanh'))
        self.conv_list.append(ConvBNBlock(
            postnet_dim, n_mel_channels, kernel_size=5, activation=None))
        self.convs = nn.Sequential(*self.conv_list)

    def forward(self, x):
        x = self.convs(x)
        return x


class Model(nn.Module):
    def __init__(self, n_vocab, d_model, embedding_dim, n_mel_channels, n_layers, n_head, d_ff, dropout):
        super(Model, self).__init__()
        self.encoder = Encoder(n_vocab, d_model, embedding_dim,
                               n_layers, n_head, d_ff, dropout)
        self.decoder = Decoder(
            n_vocab, d_model, n_mel_channels, n_layers, n_head, d_ff, dropout)
        self.Projection = Linear(d_model, n_mel_channels)
        self.Postnet = Postnet(d_model, n_mel_channels)

    def forward(self, symbols, mels):
        enc_outputs, enc_attentions, enc_padding_mask = self.encoder(symbols)
        decoder_outputs, dec_attentions, context_attentions = self.decoder(
            mels, enc_outputs, enc_padding_mask)
        mel_out = self.Projection(
            decoder_outputs.transpose(0, 1)).transpose(1, 2)
        post_out = self.Postnet(mel_out)
        mel_post_out = mel_out + post_out

        return mel_out, mel_post_out

