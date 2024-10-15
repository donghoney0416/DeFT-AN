import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.autograd import Variable
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


class DenseBlock(nn.Module):
    def __init__(self, in_channels, depth):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.block = nn.ModuleList([])
        for i in range(self.depth):
            self.block.append(nn.Sequential(
                nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1)),
                nn.LayerNorm(257),
                nn.PReLU(in_channels)
            ))

    def forward(self, input):
        skip = input
        for i in range(self.depth):
            output = self.block[i](skip)
            skip = torch.cat([output, skip], dim=1)

        return output


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        dim_head = dim
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.dropout = dropout

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h = self.heads)
        # FlashAttention
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=self.dropout)
        out = rearrange(out.transpose(1, 2), 'b h n d -> b n (h d)')
        return self.to_out(out)


class Freq_Conformer(nn.Module):
   def __init__(self, dim_model, num_head, dropout):
       super(Freq_Conformer, self).__init__()
       self.dim_model = dim_model
       self.num_head = num_head
       self.dim_ffn = 4 * self.dim_model
       self.dropout = dropout

       self.self_attn = Attention(dim=dim_model, heads=num_head, dropout=dropout)
       self.dropout1 = nn.Dropout(self.dropout)
       self.norm1 = nn.LayerNorm(self.dim_model, eps=1e-5)

       self.linear1 = nn.Linear(self.dim_model, self.dim_ffn)
       self.activation = nn.GELU()
       self.linear2 = nn.Linear(self.dim_ffn, self.dim_model)

       self.dropout3 = nn.Dropout(self.dropout)
       self.norm2 = nn.LayerNorm(self.dim_model, eps=1e-5)

   def forward(self, input):
       att_out = self.self_attn(input)
       norm_out = self.norm1(input + self.dropout1(att_out))

       ffw_out = self.linear2(self.activation(self.linear1(norm_out)))
       output = self.norm2(norm_out + self.dropout3(ffw_out))

       return output


class Time_Conformer(nn.Module):
    def __init__(self, dim_model, num_head, dropout, num_layers):
        super(Time_Conformer, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_ffn = 4 * self.dim_model
        self.dropout = dropout
        self.num_layers = num_layers

        self.self_attn = Attention(dim=dim_model, heads=num_head, dropout=dropout)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.dim_model, eps=1e-5)

        self.linear1 = nn.Linear(self.dim_model, self.dim_ffn)
        self.activation = nn.GELU()
        convs = []
        for i in range(self.num_layers):
            convs.append(nn.Sequential(
                nn.Conv1d(self.dim_ffn, self.dim_ffn, kernel_size=3, dilation = 2 ** i, padding='same', groups=self.dim_ffn),
                nn.GroupNorm(1, self.dim_ffn),
                nn.PReLU(self.dim_ffn)
            ))
        self.seq_conv = nn.Sequential(*convs)
        self.linear2 = nn.Linear(self.dim_ffn, self.dim_model)

        self.dropout3 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.dim_model, eps=1e-5)

    def forward(self, input):
        att_out = self.self_attn(input)
        norm_out = self.norm1(input + self.dropout1(att_out))

        ffw_mid = self.activation(self.linear1(norm_out)).transpose(1, 2).contiguous()
        ffw_out = self.linear2(self.seq_conv(ffw_mid).transpose(1, 2).contiguous())
        output = self.norm2(norm_out + self.dropout3(ffw_out))

        return output


class proposed_block(nn.Module):
    def __init__(self, ch_dim, num_head, dropout, depth):
        super(proposed_block, self).__init__()
        self.dense_block = DenseBlock(in_channels=ch_dim, depth=depth)
        self.freq_conformer = Freq_Conformer(dim_model=ch_dim, num_head=num_head, dropout=dropout)
        self.temp_conformer = Time_Conformer(dim_model=ch_dim, num_head=num_head, dropout=dropout, num_layers=3)

    def forward(self, input):
        B, C, L, F = input.size()

        output_c = self.dense_block(input)                          # B, C, L, F

        input_f = output_c.permute(0, 2, 3, 1).contiguous()         # B, L, F, C
        input_f = input_f.view(B*L, F, C)                           # B*L, F, C
        output_f = self.freq_conformer(input_f)                     # B*L, F, C
        output_f = output_f.view(B, L, F, C)                        # B, L, F, C

        input_t = output_f.permute(0, 2, 1, 3).contiguous()         # B, F, L, C
        input_t = input_t.view(B*F, L, C)                           # B*F, L, C
        output_t = self.temp_conformer(input_t)                     # B*F, L, C
        output_t = output_t.view(B, F, L, C)                        # B, F, L, C

        output = output_t.permute(0, 3, 2, 1)                       # B, C, L, F

        return output


class Network(nn.Module):
    def __init__(self, mic_num=4, ch_dim=64, win=512, num_head=4, dropout=0.1, num_layer=4, depth=5):
        super(Network, self).__init__()

        self.mic_num = mic_num
        self.out_ch_num = out_ch_num
        self.ch_dim = ch_dim

        self.win = win
        self.hop = self.win // 2

        self.prelu = nn.PReLU()
        self.dim_model = ch_dim
        self.num_head = num_head
        self.dim_ffn = self.dim_model * 2

        self.dropout = dropout
        self.n_conv_groups = self.dim_ffn

        self.num_layer = num_layer

        self.inp_conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.mic_num, out_channels=self.ch_dim, kernel_size=(5, 5), padding=(2, 2)),
            nn.LayerNorm(257),
            nn.PReLU(self.ch_dim)
        )

        self.proposed_block = nn.ModuleList([])
        for ii in range(num_layer):
            self.proposed_block.append(
                proposed_block(ch_dim=ch_dim, num_head=num_head, dropout=dropout, depth=depth)
            )

        self.out_conv = nn.Conv2d(in_channels=self.ch_dim, out_channels=2 * self.mic_num, kernel_size=(5, 5), padding=(2, 2))

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.hop + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.hop)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input):
        input, rest = self.pad_signal(input)
        B, M, T = input.size()                                  # batch B, mic M, time samples T

        stft_input = torch.stft(input.view([-1, T]), n_fft=self.win, hop_length=self.hop, return_complex=False)
        _, F, L, _ = stft_input.size()                          # B*M , F= num freqs, L= num frame, 2= real imag
        xi = stft_input.view([B, M, F, L, 2])                   # B*M, F, L, 2 -> B, M, F, L, 2
        xi = xi.permute(0, 1, 4, 3, 2).contiguous()             # B, M, 2, L, F
        xi = xi.view([B, M*2, L, F])                            # B, 2*M, L, F

        xo = self.inp_conv(xi)                                  # B, C, L, F
        for idx in range(self.num_layer):
            xo = self.proposed_block[idx](xo)                   # B, C, L, F
        mask = self.out_conv(xo)                                # B, 2*M, L, F

        masked_enc_out = xi * mask
        yo = masked_enc_out.permute(0, 3, 2, 1).contiguous()    # B, 2M, L, F -> B, F, L, 2M
        yo = yo.reshape([B, F, L, 2, M]).permute(0, 4, 1, 2, 3).reshape([B*M, F, L, 2])                        # BM, F, L, 2

        output = torch.istft(torch.complex(yo[:, :, :, 0], yo[:, :, :, 1]), n_fft=self.win, hop_length=self.hop, return_complex=False)
        output = output[:, self.hop:-(rest + self.hop)].reshape([B, M, -1])

        return output