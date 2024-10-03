import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import logging
import torch.nn.functional as F

logging.getLogger("transformers").setLevel(logging.ERROR)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class RMSCNN(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(RMSCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type='max'):

        # x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            return x

        return x


class ResMultiConv(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''

    def __init__(self, channels=16, **kwargs):
        super(ResMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels * 2)

    def forward(self, x):
        x3 = self.conv3(x) + x
        x5 = self.conv5(x) + x
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ResMultiConv_1D_down(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''

    def __init__(self, channels=16, **kwargs):
        super(ResMultiConv_1D_down, self).__init__()
        self.conv3 = nn.Conv1d(kernel_size=3, in_channels=channels, out_channels=channels//4, padding=1)
        self.conv5 = nn.Conv1d(kernel_size=5, in_channels=channels, out_channels=channels//4, padding=2)
        self.bn = nn.BatchNorm1d(channels // 2)
        self.reduce_conv = nn.Conv1d(in_channels=channels, out_channels=channels // 4, kernel_size=1)
    def forward(self, x):
        x_reduce = self.reduce_conv(x)
        x3 = self.conv3(x) + x_reduce
        x5 = self.conv5(x) + x_reduce
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ResMultiConv_1D_nomal(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''

    def __init__(self, channels=16, **kwargs):
        super(ResMultiConv_1D_nomal, self).__init__()
        self.conv3 = nn.Conv1d(kernel_size=3, in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv1d(kernel_size=5, in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm1d(channels * 2)
        self.reduce_conv = nn.Conv1d(in_channels=channels * 2, out_channels=channels, kernel_size=1)
    def forward(self, x):
        x3 = self.conv3(x) + x
        x5 = self.conv5(x) + x
        x = torch.cat((x3, x5), 1)
        x = self.bn(x)
        x = F.relu(x)
        x = self.reduce_conv(x)
        return x

class CustomWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super(CustomWav2Vec2Model, self).__init__(config)
        # Replace the CNN layers with RMSCNN
        self.feature_extractor = RMSCNN(in_channels=config.hidden_size, out_channels=config.hidden_size)

    def forward(self, input_values):
        # Extract features from input values using RMSCNN
        x = self.feature_extractor(input_values)
        # Continue with the rest of the Wav2Vec2 forward pass
        return super(CustomWav2Vec2Model, self).forward(x)


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=3, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=3, out_channels=16, padding=(0, 1))
        self.maxp = nn.MaxPool2d(kernel_size=(1, 2))
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.atten_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.rmscnn = RMSCNN(config.hidden_size)
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False
        self.init_weights()

    def init_weight(self):
        init_bn(self.bn0)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_bn(self.bn5)
        # init_layer(self.fc)
        # init_layer(self.fc1)
        # init_layer(self.fc2)
        # init_layer(self.fc3)

    def forward(
            self,
            input_values,
            attention_mask,
    ):

        bs, c, f, t = input_values.shape

        input = input_values.transpose(2, 3)  # (bs,c,t,f)
        input = input.transpose(1, 3)  # (bs,f,t,c)
        input = self.bn0(input)
        input = input.transpose(1, 3)  # (bs,c,t,f)

        xa = self.conv1a(input)  # (bs, 16, ?, 64)
        xa = self.bn1a(xa)  # (bs, 16, ?, 64)
        xa = F.relu(xa)

        xb = self.conv1b(input)  # (bs, 16, ?, 64)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        e_attention_mask = torch.repeat_interleave(attention_mask, 2, dim=1)
        x = torch.cat((xa, xb), 2)  # (32, 16, 2?, 64)

        x = self.conv2(x)  # (32, 32, 2?, 64)
        x = self.maxp1(x)  # (32, 32, ?, 32)
        attention_mask = e_attention_mask.unsqueeze(1).float()
        attention_mask = self.atten_pool(attention_mask)

        x = self.conv3(x)  # (32, 64, ?, 32)
        x = self.maxp1(x)  # (32, 64, ?/2, 16)
        # attention_mask = attention_mask.unsqueeze(1)
        attention_mask = self.atten_pool(attention_mask)

        x = self.conv4(x)  # (32, 128, ?/2, 16)
        x = self.maxp1(x)  # (32, 128, ?/4, 8)
        # attention_mask = attention_mask.unsqueeze(1)
        attention_mask = self.atten_pool(attention_mask).squeeze(1).bool()

        x = self.conv5(x)  # (32, 128, ?/4, 8)
        x = self.maxp(x)  # (32, 128, ?/8, 4)
        # attention_mask = attention_mask.unsqueeze(1)
        # attention_mask = self.atten_pool(attention_mask).squeeze(1).bool()
        x = self.bn5(x)
        x = F.relu(x)
        input_values = x.permute(0,2,1,3).flatten(start_dim=2)  #.permute(0,2,1)   #(32, ?/8, 512) ->(32, 512, ?/8)

        attention_mask = None
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states = hidden_states * torch.reshape(attention_mask, (-1, attention_mask.shape[-1], 1))
            hidden_states = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            hidden_states = hidden_states / torch.reshape(attention_sum, (-1, 1))
        else:
            hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

class EmotionModel_RMSCNN(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        # self.wav2vec2 = Wav2Vec2Model(config)
        # self.classifier = RegressionHead(config)
        self.bn0 = nn.BatchNorm2d(40)
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=3, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=3, out_channels=16, padding=(0, 1))
        self.maxp = nn.MaxPool2d(kernel_size=(1, 2))
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256, 128, bias=True)
        self.fc1 = nn.Linear(128, 1, bias=True)
        # self.fc2 = nn.Linear(256, 1, bias=True)
        # self.rmscnn = RMSCNN(config.hidden_size)
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False
        self.init_weights()

    def init_weight(self):
        init_bn(self.bn0)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_layer(self.fc)
        init_layer(self.fc1)
        # init_layer(self.fc2)
        # init_layer(self.fc3)

    def forward(
            self,
            input_spectrogram,
            attention_mask,
    ):

        bs, c, f, t = input_spectrogram.shape

        input = input_spectrogram.transpose(2, 3)  # (bs,c,t,f)
        input = input.transpose(1, 3)  # (bs,f,t,c)
        input = self.bn0(input)
        input = input.transpose(1, 3)  # (bs,c,t,f)

        xa = self.conv1a(input)  # (bs, 16, ?, 64)
        xa = self.bn1a(xa)  # (bs, 16, ?, 64)
        xa = F.relu(xa)

        xb = self.conv1b(input)  # (bs, 16, ?, 64)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 2?, 64)

        x = self.conv2(x)  # (32, 32, 2?, 64)
        x = self.maxp1(x)  # (32, 32, ?, 32)

        x = self.conv3(x)  # (32, 64, ?, 32)
        x = self.maxp(x)  # (32, 64, ?/2, 16)

        x = self.conv4(x)  # (32, 128, ?/2, 16)
        x = self.maxp(x)  # (32, 128, ?/4, 8)

        x = self.conv5(x)  # (32, 128, ?/4, 8)
        x = self.maxp(x)  # (32, 128, ?/8, 4)
        # # attention_mask = attention_mask.unsqueeze(1)
        # attention_mask = self.atten_pool(attention_mask).squeeze(1).bool()
        x = self.bn5(x)
        x = F.relu(x)

        avg_out = self.global_avg_pool(x)
        max_out = self.global_max_pool(x)
        x = torch.cat((avg_out, max_out), dim=1)
        x = x.view(x.size(0), -1)
        x_rms = self.fc(x)
        logits = self.fc1(x_rms)

        return logits


class EmotionModel_Concat(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.bn0 = nn.BatchNorm2d(40)
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=3, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=3, out_channels=16, padding=(0, 1))
        self.maxp = nn.MaxPool2d(kernel_size=(1, 2))
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256, 128, bias=True)
        self.fc1 = nn.Linear(1024, 128, bias=True)
        self.fc2 = nn.Linear(256, 1, bias=True)
        # self.rmscnn = RMSCNN(config.hidden_size)
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False
        self.init_weights()

    def init_weight(self):
        init_bn(self.bn0)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_layer(self.fc)
        init_layer(self.fc1)
        init_layer(self.fc2)
        # init_layer(self.fc3)

    def forward(
            self,
            input_values, attention_mask,
            input_spectrogram,
    ):

        bs, c, f, t = input_spectrogram.shape

        input = input_spectrogram.transpose(2, 3)  # (bs,c,t,f)
        input = input.transpose(1, 3)  # (bs,f,t,c)
        input = self.bn0(input)
        input = input.transpose(1, 3)  # (bs,c,t,f)

        xa = self.conv1a(input)  # (bs, 16, ?, 64)
        xa = self.bn1a(xa)  # (bs, 16, ?, 64)
        xa = F.relu(xa)

        xb = self.conv1b(input)  # (bs, 16, ?, 64)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 2?, 64)

        x = self.conv2(x)  # (32, 32, 2?, 64)
        x = self.maxp1(x)  # (32, 32, ?, 32)

        x = self.conv3(x)  # (32, 64, ?, 32)
        x = self.maxp(x)  # (32, 64, ?/2, 16)

        x = self.conv4(x)  # (32, 128, ?/2, 16)
        x = self.maxp(x)  # (32, 128, ?/4, 8)

        x = self.conv5(x)  # (32, 128, ?/4, 8)
        x = self.maxp(x)  # (32, 128, ?/8, 4)
        # # attention_mask = attention_mask.unsqueeze(1)
        # attention_mask = self.atten_pool(attention_mask).squeeze(1).bool()
        x = self.bn5(x)
        x = F.relu(x)

        avg_out = self.global_avg_pool(x)
        max_out = self.global_max_pool(x)
        x = torch.cat((avg_out, max_out), dim=1)
        x = x.view(x.size(0), -1)
        x_rms = self.fc(x)
        # ------------------------------
        # input_values = x.permute(0,2,1,3).flatten(start_dim=2)  #.permute(0,2,1)   #(32, ?/8, 512) ->(32, 512, ?/8)

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states = hidden_states * torch.reshape(attention_mask, (-1, attention_mask.shape[-1], 1))
            hidden_states = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            hidden_states = hidden_states / torch.reshape(attention_sum, (-1, 1))
        else:
            hidden_states = torch.mean(hidden_states, dim=1)
        x_w2v2 = self.fc1(hidden_states)
        x_all = torch.cat((x_rms, x_w2v2), dim=1)
        logits = self.fc2(x_all)

        return logits

class EmotionModel_Concat_pretrain(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.bn0 = nn.BatchNorm2d(40)
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=3, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=3, out_channels=16, padding=(0, 1))
        self.maxp = nn.MaxPool2d(kernel_size=(1, 2))
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256, 128, bias=True)
        self.fc1 = nn.Linear(128, 1, bias=True)
        self.fc3 = nn.Linear(1024, 128, bias=True)
        self.fc4 = nn.Linear(256, 1, bias=True)
        # self.rmscnn = RMSCNN(config.hidden_size)
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False
        self.init_weights()
    def init_weight(self):
        init_bn(self.bn0)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_layer(self.fc)
        init_layer(self.fc3)
        init_layer(self.fc4)
        # init_layer(self.fc3)

    def forward(
            self,
            input_values, attention_mask,
            input_spectrogram,
    ):

        input = input_spectrogram.transpose(2, 3)  # (bs,c,t,f)
        input = input.transpose(1, 3)  # (bs,f,t,c)
        input = self.bn0(input)
        input = input.transpose(1, 3)  # (bs,c,t,f)

        xa = self.conv1a(input)  # (bs, 16, ?, 64)
        xa = self.bn1a(xa)  # (bs, 16, ?, 64)
        xa = F.relu(xa)

        xb = self.conv1b(input)  # (bs, 16, ?, 64)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)  # (32, 16, 2?, 64)

        x = self.conv2(x)  # (32, 32, 2?, 64)
        x = self.maxp1(x)  # (32, 32, ?, 32)

        x = self.conv3(x)  # (32, 64, ?, 32)
        x = self.maxp(x)  # (32, 64, ?/2, 16)

        x = self.conv4(x)  # (32, 128, ?/2, 16)
        x = self.maxp(x)  # (32, 128, ?/4, 8)

        x = self.conv5(x)  # (32, 128, ?/4, 8)
        x = self.maxp(x)  # (32, 128, ?/8, 4)
        # # attention_mask = attention_mask.unsqueeze(1)
        # attention_mask = self.atten_pool(attention_mask).squeeze(1).bool()
        x = self.bn5(x)
        x = F.relu(x)

        avg_out = self.global_avg_pool(x)
        max_out = self.global_max_pool(x)
        x = torch.cat((avg_out, max_out), dim=1)
        x = x.view(x.size(0), -1)
        x_rms = self.fc(x)
        # ------------------------------
        # input_values = x.permute(0,2,1,3).flatten(start_dim=2)  #.permute(0,2,1)   #(32, ?/8, 512) ->(32, 512, ?/8)

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states = hidden_states * torch.reshape(attention_mask, (-1, attention_mask.shape[-1], 1))
            hidden_states = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            hidden_states = hidden_states / torch.reshape(attention_sum, (-1, 1))
        else:
            hidden_states = torch.mean(hidden_states, dim=1)
        x_w2v2 = self.fc3(hidden_states)
        x_all = torch.cat((x_rms, x_w2v2), dim=1)
        logits = self.fc4(x_all)

        return logits

class EmotionModel_w2v2rmsc_1D(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
        # self.conv1 = ResMultiConv_1D_down(1024)
        # self.conv2 = ResMultiConv_1D_down(512)
        self.conv1 = ResMultiConv_1D_nomal(1024)
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False

    def forward(
            self,
            input_values,
            attention_mask,
    ):

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = hidden_states.permute(0,2,1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = hidden_states.permute(0,2,1)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states = hidden_states * torch.reshape(attention_mask, (-1, attention_mask.shape[-1], 1))
            hidden_states = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            hidden_states = hidden_states / torch.reshape(attention_sum, (-1, 1))
        else:
            hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


class EmotionModel_w2v2rmsc_1D_down(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
        # self.conv1 = ResMultiConv_1D_down(1024)
        # self.conv2 = ResMultiConv_1D_down(512)
        self.conv1 = ResMultiConv_1D_down(1024)
        self.conv2 = ResMultiConv_1D_down(512)
        self.fc1 = nn.Linear(256, 1, bias=True)
        # for param in self.wav2vec2.parameters():
        #     param.requires_grad = False

    def init_weight(self):
        init_layer(self.fc1)

    def forward(
            self,
            input_values,
            attention_mask,
    ):

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = hidden_states.permute(0,2,1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.permute(0,2,1)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states = hidden_states * torch.reshape(attention_mask, (-1, attention_mask.shape[-1], 1))
            hidden_states = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            hidden_states = hidden_states / torch.reshape(attention_sum, (-1, 1))
        else:
            hidden_states = torch.mean(hidden_states, dim=1)
        # logits = self.classifier(hidden_states)
        logits = self.fc1(hidden_states)
        return hidden_states, logits


