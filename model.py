import torch
from torch import nn
import math

from torch.nn.modules.conv import ConvTranspose2d
from torch.utils import data


def nc2nlc(a):
    bs, c = a.size()
    row = column = int(math.sqrt(c))
    assert(row * column == c)
    matrixa = a.view(bs, row, column)
    b = torch.rand(bs,row,column*2, device=a.device)
    for i in range(row):
        b[:,i,:] = torch.cat([matrixa[:,i,:],matrixa[:,:,i]],dim=1)
    return b


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, dims, dropout_ratio=0.0, activation='relu'):
        super().__init__()

        if activation == "relu":
            self.activation_func = nn.ReLU()
        else:
            self.activation_func = nn.Tanh()

        self.ae_blocks = []
        self.networks = nn.Sequential()
        for i in range(len(dims) - 2):
            self.ae_blocks.append(nn.Linear(dims[i], dims[i+1]))
            self.ae_blocks.append(nn.BatchNorm1d(dims[i+1]))
            self.ae_blocks.append(self.activation_func)
            self.ae_blocks.append(nn.Dropout(dropout_ratio))

        self.ae_seq = nn.Sequential(*self.ae_blocks)
        self.output_block = nn.Sequential(
            nn.Linear(dims[-2], dims[-1]),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.ae_seq(x)
        x = self.output_block(x)
        return x



class DAEM(nn.Module):
    def __init__(self, dims, dropout_ratio=0.0, activation='relu'):
        super().__init__()

        if activation == "relu":
            self.activation_func = nn.ReLU()
        else:
            self.activation_func = nn.Tanh()

        self.W = nn.Parameter(torch.Tensor(1, dims[0]))
        nn.init.uniform_(self.W, a=-0.0, b=1.0) 
        self.ae_blocks = []
        self.networks = nn.Sequential()
        for i in range(len(dims) - 2):
            self.ae_blocks.append(nn.Linear(dims[i], dims[i+1]))
            self.ae_blocks.append(nn.BatchNorm1d(dims[i+1]))
            self.ae_blocks.append(self.activation_func)
            self.ae_blocks.append(nn.Dropout(dropout_ratio))

        self.ae_seq = nn.Sequential(*self.ae_blocks)
        self.output_block = nn.Sequential(
            nn.Linear(dims[-2], dims[-1]),
        )

    def forward(self, x, mask):
        x = x + self.W * mask
        x = self.ae_seq(x)
        x = self.output_block(x)
        return x


class Discriminator(nn.Module):
    # input is (N, C)
    def __init__(self, input, hid, activation='relu'):
        super().__init__()

        if activation == "relu":
            self.activation_func = nn.ReLU()
        else:
            self.activation_func = nn.Tanh()

        # using nn.utils.parametrizations.spectral_norm causes problem and I do not know why !!!

        self.l1 = nn.utils.spectral_norm(nn.Linear(input, hid))
        self.l2 = nn.utils.spectral_norm(nn.Linear(hid, hid//2))
        # self.out  = nn.Linear(hid, 1)
        self.l3 = nn.Linear(hid//2,hid//4)
        nn.init.xavier_uniform_(self.l3.weight.data)
        self.l3  = nn.utils.spectral_norm(self.l3)
        # output layer should use spectral norm or it won't train
        self.out = nn.Linear(hid//4,1)
        nn.init.xavier_uniform_(self.out.weight.data)
        self.out  = nn.utils.spectral_norm(self.out)


    def forward(self, x):
        x = self.l1(x)
        x = self.activation_func(x)
        x = self.l2(x)
        x = self.activation_func(x)
        x = self.l3(x)
        x = self.activation_func(x)
        x = self.out(x)
        x = torch.nn.Sigmoid()(x)
        return x



class CNN(nn.Module):
    def __init__(self, dataname, activation='relu'):
        super().__init__()

        if activation == "relu":
            self.activation_func = nn.ReLU()
        else:
            self.activation_func = nn.Tanh()

        if dataname == "ab": # 12
            self.networks = nn.Sequential(
                nn.Conv2d(2,4,3), # 10
                self.activation_func,
                nn.Conv2d(4,8,3), # 8
                self.activation_func,
                nn.Conv2d(8,16,3), # 6
                self.activation_func,
                nn.Conv2d(16,32,3), # 4
                self.activation_func,
                nn.Conv2d(32,64,3), # 2
                self.activation_func,
                nn.ConvTranspose2d(64,32,3), # 4
                self.activation_func,
                nn.ConvTranspose2d(32,16,3), # 6
                self.activation_func,
                nn.ConvTranspose2d(16,8,3), # 8
                self.activation_func,
                nn.ConvTranspose2d(8,4,3), # 10
                self.activation_func,
                nn.ConvTranspose2d(4,1,3) # 12
            )
        elif dataname == "se":
            self.networks = nn.Sequential(
                # 99
                nn.Conv2d(2,4,5, stride=2), # 48
                self.activation_func,
                nn.Conv2d(4,8,5, stride=2), # 22
                self.activation_func,
                nn.Conv2d(8,16,5, stride=2), # 9
                self.activation_func,
                nn.Conv2d(16,32,3, stride=2), # 4
                self.activation_func,
                nn.Conv2d(32,64,3), # 2
                self.activation_func,
                nn.ConvTranspose2d(64,32,3), # 4
                self.activation_func,
                nn.ConvTranspose2d(32,16,3, stride=2), # 9
                self.activation_func,
                nn.ConvTranspose2d(16,8,5, stride=2, output_padding=1), # 22
                self.activation_func,
                nn.ConvTranspose2d(8,4,5, stride=2, output_padding=1), # 48
                self.activation_func,
                nn.ConvTranspose2d(4,1,5, stride=2), # 99
                self.activation_func,
            )


        else: 
            assert(dataname == "ge") # 23
            self.networks = nn.Sequential(
                nn.Conv2d(2,4,5), # 19
                self.activation_func,
                nn.Conv2d(4,8,5), # 15
                self.activation_func,
                nn.Conv2d(8,16,3, stride=2), # 7
                self.activation_func,
                nn.Conv2d(16,32,3), # 5
                self.activation_func,
                nn.Conv2d(32,64,3), # 3
                self.activation_func,
                nn.ConvTranspose2d(64,32,3), # 5
                self.activation_func,
                nn.ConvTranspose2d(32,16,3), # 7
                self.activation_func,
                nn.ConvTranspose2d(16,8,3, stride=2), # 15
                self.activation_func,
                nn.ConvTranspose2d(8,4,5), # 19
                self.activation_func,
                nn.ConvTranspose2d(4,1,5), # 23
            )
    def forward(self, x, mask):
        x = self.networks(torch.cat((x,mask),1))
        return x



class AttentionWithPE(nn.Module):
    def __init__(self, feature_in, feature_hid, heads, num_layers, activation='relu'):
        super().__init__()

        if activation == "relu":
            self.activation_func = nn.ReLU()
        else:
            self.activation_func = nn.Tanh()

        self.rows = feature_in // 2
        self.W = nn.Parameter(torch.Tensor(1, self.rows ** 2))
        nn.init.uniform_(self.W, a=-0.0, b=1.0)
        self.pe = PositionalEncoding(feature_hid)
        #input embedding
        self.input_proj = nn.Linear(feature_in, feature_hid)
        self.encoder = nn.Sequential(
            # self-attention
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_hid, nhead=heads, \
                dim_feedforward=feature_hid*2,dropout=0.0, batch_first=True), num_layers=num_layers),
            #output
            nn.Linear(feature_hid,feature_in), # N L C
        )

    def forward(self, x, mask):
        # input is N, L2
        x = x + self.W * mask
        x = nc2nlc(x)
        x = self.activation_func(self.input_proj(x)) # B, L, D
        x = self.pe(x)
        x = self.encoder(x)
        # N, L, C to N, L, L
        x = x[:,:,:self.rows] + x[:,:,self.rows:].transpose(1,2)
        # N, L, L to N, L2
        x = nn.Flatten()(x)
        return x

    def gen_attentionmap(self, x, mask):
        x = x + self.W * mask
        x = nc2nlc(x)
        x = self.activation_func(self.input_proj(x)) # B, L, D
        x = self.pe(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(max_len, 1, d_model)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # """
        # Args:
        #     x: Tensor, shape [seq_len, batch_size, embedding_dim]
        # """
        # x = x + self.pe[:x.size(0)]
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # print(x.shape, self.pe.shape)
        x = x + self.pe[0,:x.size(1),:]
        return self.dropout(x)