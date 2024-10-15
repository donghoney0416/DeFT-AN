import numpy as np
from itertools import permutations
from torch.autograd import Variable
import torch
import torch.nn as nn
from scipy import linalg
import scipy

class stft_loss(nn.Module):
    def __init__(self, win, hop, loss_type='mae'):
        super(stft_loss, self).__init__()
        self.win = win
        self.hop = hop
        self.loss_type = loss_type

    def forward(self, est, org):
        stft_org = torch.stft(org.view([-1, org.size(2)]), n_fft=self.win, hop_length=self.hop, window=torch.hann_window(self.win).type(org.type()), return_complex=False)
        stft_est = torch.stft(est.view([-1, est.size(2)]), n_fft=self.win, hop_length=self.hop, window=torch.hann_window(self.win).type(est.type()), return_complex=False)
        if self.loss_type == 'mse':
            stft_loss = torch.mean((stft_org - stft_est) ** 2)
        elif self.loss_type == 'mae':
            stft_loss = torch.mean(torch.abs(stft_org - stft_est))

        return stft_loss


class PCM_Loss(nn.Module):
    def __init__(self):
        super(PCM_Loss, self).__init__()
        self.loss = stft_loss(win=512, hop=256, loss_type='mae')

    def forward(self, mixed, estimation, origin):
        speech_estimation = estimation
        speech_origin = origin

        noise_estimation = mixed - estimation
        noise_origin = mixed - origin

        loss_speech = self.loss(speech_estimation, speech_origin)
        loss_noise = self.loss(noise_estimation, noise_origin)

        tot_loss = 0.5*loss_speech + 0.5*loss_noise

        return tot_loss