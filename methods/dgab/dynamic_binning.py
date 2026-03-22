import torch
import torch.nn.functional as F
from rtdl_num_embeddings import compute_bins
from torch import nn
from torch.nn import Parameter


class DyPLEC(nn.Module):
    def __init__(self, d_in, n_bins, dy_raw_bin_width=True, n_heads=1):
        super().__init__()
        self.n_bins = n_bins
        self.dy_raw_bin_width = dy_raw_bin_width
        self.n_heads = n_heads
        self.raw_bin_width = nn.Parameter(
            torch.randn(d_in, self.n_heads, self.n_bins),
            requires_grad=self.dy_raw_bin_width,
        )
        self.register_buffer("mask", torch.tril(torch.ones(self.n_bins, self.n_bins)))
        self.o_d_in = d_in
        mask_left = torch.ones((d_in, self.n_bins))
        mask_left[:, -1] = 0
        self.register_buffer("mask_left", mask_left.bool())
        mask_right = torch.ones((d_in, self.n_bins))
        mask_right[:, 0] = 0
        self.register_buffer("mask_right", mask_right.bool())
        self.d_out = self.o_d_in * self.n_heads * self.n_bins

    def forward(self, inp):
        bin_width = self.raw_bin_width.softmax(dim=-1)
        bin_axis = (bin_width[:, :, None, :] * self.mask[None, None, :, :]).sum(dim=-1)
        zeros = torch.zeros(
            (bin_axis.shape[0], self.n_heads, 1), device=bin_width.device
        )
        new_bin_axis = torch.cat((zeros, bin_axis), dim=-1)[..., : self.n_bins]
        diff = inp[:, :, None, None] - new_bin_axis
        rate = diff / bin_width
        rate = rate.transpose(1, 2).flatten(-2, -1)
        rate[:, :, self.mask_left.flatten()] = 1 - F.relu(
            1 - rate[:, :, self.mask_left.flatten()]
        )
        rate[:, :, self.mask_right.flatten()] = F.relu(
            rate[:, :, self.mask_right.flatten()]
        )
        x = rate.view(-1, self.o_d_in, self.n_bins)
        return x

    def init_params(self, x, y):
        n_bins = self.n_bins
        mask_left = torch.ones((self.o_d_in, self.n_bins))
        mask_right = torch.ones((self.o_d_in, self.n_bins))
        bins = compute_bins(
            x,
            n_bins=n_bins,
            tree_kwargs={"min_samples_leaf": int(float(len(y)) * 0.005) + 1},
            y=y,
            regression=False,
        )
        print(bins)
        bins_matrix = torch.zeros((x.shape[1], n_bins + 1))
        for i, bin in enumerate(bins):
            l = len(bin)
            bins_matrix[i, -l:] = bin
            mask_left[i, -1] = 0
            mask_right[i, 1 - l] = 0
        logs = (bins_matrix.diff() + 1e-8).log()
        s = -torch.mean(logs)
        raw_bin_width = logs + s
        self.raw_bin_width.data.copy_(raw_bin_width.unsqueeze(1))
        self.mask_left.data.copy_(mask_left)
        self.mask_right.data.copy_(mask_right)
        print("dyple in and out:", self.o_d_in, self.d_out)


class NLinear(nn.Module):
    def __init__(self, n: int, in_features: int, out_features: int, bias=True) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.b = bias
        if self.b:
            self.bias = Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.b:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        if self.b:
            x = x + self.bias
        return x


class DyPLEM(nn.Module):
    def __init__(
        self,
        d_in,
        d_feat_emb,
        d_hidden,
        n_bins,
        dy_raw_bin_width,
        pre_use_bn,
        pre_dropout,
        use_feat_emb=False,
    ):
        super().__init__()
        self.fc_in = nn.Sequential()
        self.drop = nn.Dropout(pre_dropout)
        self.dy_ple_layer = DyPLEC(d_in, n_bins, dy_raw_bin_width, 1)
        self.nlinear = NLinear(d_in, n_bins, d_feat_emb)
        self.bn0 = nn.BatchNorm1d(d_in) if pre_use_bn else nn.Identity()
        self.linear = nn.Linear(d_in * d_feat_emb, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden) if pre_use_bn else nn.Identity()
        self.use_feat_emb = use_feat_emb

    def forward(self, h):
        h = self.dy_ple_layer(h)
        h = self.nlinear(h)
        feat_h = h = F.relu(h, inplace=True)
        h = self.bn0(h)
        h = self.drop(h)
        h = h.flatten(start_dim=1)
        h = self.linear(h)
        h = F.relu(h, inplace=True)
        h = self.bn1(h)
        if self.use_feat_emb:
            return h, feat_h
        else:
            return h

    def init_params(self, x, y):
        self.dy_ple_layer.init_params(x, y)
