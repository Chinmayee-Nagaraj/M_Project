import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST #TODO
from torchvision.transforms import ToTensor
# removed: from torchvision.ops.misc import MLP #TODO

from tqdm import tqdm

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional


class GRUFFNBlock(nn.Module):
    """
    GRU-based position-aware feed-forward block.
    Implements: Res = Linear(ReLU(GRU(Mid)))
    Input/Output shape: (batch, seq_length, in_dim)
    - gru_hidden_size should be gru_dim (e.g. d*4)
    """

    def __init__(self, in_dim: int, gru_hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru = nn.GRU(input_size=in_dim, hidden_size=gru_hidden_size, batch_first=True, bidirectional=False)
        self.act = nn.ReLU()
        self.proj = nn.Linear(gru_hidden_size, in_dim)
        self.dropout = nn.Dropout(dropout)

        # weight init #TODO #check the functionality and use of the below code
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_length, in_dim)
        returns: (batch, seq_length, in_dim)
        """
        # GRU expects (batch, seq, feat) -> returns output (batch, seq, hidden)
        gru_out, _ = self.gru(x)  # gru_out: (batch, seq_length, gru_hidden_size)
        activated = self.act(gru_out)
        out = self.proj(activated)  # (batch, seq_length, in_dim)
        out = self.dropout(out)
        return out


class EncoderBlock(nn.Module):  
    """Transformer encoder block with GRU-based FFN."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        gru_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # GRU-FFN block (replaces MLPBlock)
        self.ln_2 = norm_layer(hidden_dim)
        # gru_dim is used as the GRU hidden size (e.g., hidden_dim * 4)
        self.gru_ffn = GRUFFNBlock(hidden_dim, gru_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x_attn, _ = self.self_attention(x, x, x, need_weights=False)
        x_attn = self.dropout(x_attn)
        x = x + x_attn
        
        #might have to make it pre LN if it works better #TODO
        #y = self.ln_2(x)
        #y = self.gru_ffn(y)

        x = self.ln_2(x)
        y = self.gru_ffn(x)

        return x + y


class Encoder(nn.Module):   #TODO-- Remove the redundancy, and fix num_layer=1 a constant
    """Improved Transformer Model Encoder"""

    def __init__(
        self,
        seq_length: int,
        num_layers: int, #make default = 1
        num_heads: int,
        hidden_dim: int,
        gru_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):

        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        
        #NOT NEEDED #TODO
        #position = torch.arange(seq_length).unsqueeze(1)  # (seq_length, 1)
        #div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))  # (hidden_dim/2,)

        #pe = torch.zeros(seq_length, hidden_dim)
        #pe[:, 0::2] = torch.sin(position * div_term)
        #pe[:, 1::2] = torch.cos(position * div_term)

        #pe = pe.unsqueeze(0)  # (1, seq_length, hidden_dim)
        #self.pos_embedding = nn.Parameter(pe)

        self.dropout = nn.Dropout(dropout)

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                gru_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )

        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        #input = input + self.pos_embedding 
        return self.ln(self.layers(self.dropout(input)))


class Improved_Transformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,  # number of encoder blocks
        num_heads: int,  # heads in multi-head attention
        hidden_dim: int,  # embedding size
        gru_dim: int,  # gru hidden size (recommend: hidden_dim * 4)
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        #num_classes: int = 10, #TODO
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs=None,
    ):

        super().__init__()

        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        #self.num_classes = num_classes #TODO

        # To get embedding projection from image 
        self.conv_proj = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        seq_length = (image_size // patch_size) ** 2

        # Add a class token #NOT NEEDED #TODO
        #self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        #seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            gru_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        self.seq_length = seq_length

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        #n = x.shape[0]

        # Expand the class token to the full batch #NOT NEEDED #TODO
        #batch_class_token = self.class_token.expand(n, -1, -1)
        #x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        #x = x[:, 0]

        #x = self.heads(x)

        return x


if __name__ == "__main__":
    batch_size = 2
    image_size = 32
    patch_size = 8
    hidden_dim = 64
    gru_dim = 128
    num_heads = 4
    num_layers = 1

    model = Improved_Transformer(image_size, patch_size,
                                 num_layers, num_heads,
                                 hidden_dim, gru_dim)

    dummy = torch.randn(batch_size, 1, image_size, image_size)
    out = model(dummy)

    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)  # (batch, seq_len, hidden)


'''
def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root="datasets", train=True, download=True, transform=transform)
    test_set = MNIST(root="datasets", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    model = Improved_Transformer(
        image_size=28,
        patch_size=4,
        num_layers=1,
        num_heads=4, #8,
        hidden_dim= 24, #48, #embedding size
        gru_dim= 32, #64,
        dropout = 0.0,
        attention_dropout = 0.0,
        #num_classes = 10,
        representation_size = None,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs = None
    ).to(device)

    load_checkpoint = False
    save_every = 5
    START_EPOCH = 0  #change load_checkpoint
    N_EPOCHS = 10
    LR = 0.005
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Training loop
    print("\nTraining: \n")

    #load checkpoint
    if load_checkpoint:
        file = f'checkpoints/checkpoint_{START_EPOCH}.pth'
        checkpoint = torch.load(file)

        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(START_EPOCH, START_EPOCH + N_EPOCHS):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        #saving checkpoints
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            filename = f'checkpoints/checkpoint_{epoch+1}.pth'

            print("=> Saving checkpoint")
            torch.save(checkpoint, filename)
            
    # Test loop
    print("\nTesting: \n")
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
'''
