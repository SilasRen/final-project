import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device=None):
        h, w = spatial_size
        shape = (batch_size, self.hidden_dim, h, w)
        h0 = torch.zeros(shape, device=device)
        c0 = torch.zeros(shape, device=device)
        return h0, c0


class ConvLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.layers.append(
                ConvLSTMCell(dims[i], dims[i + 1], kernel_size)
            )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        device = x.device

        h_t = []
        c_t = []
        for l in range(self.num_layers):
            h, c = self.layers[l].init_hidden(
                batch_size=B,
                spatial_size=(H, W),
                device=device,
            )
            h_t.append(h)
            c_t.append(c)

        for t in range(T):
            x_t = x[:, t]  # [B, C, H, W]
            for l in range(self.num_layers):
                cell = self.layers[l]
                h, c = cell(x_t, h_t[l], c_t[l])
                h_t[l], c_t[l] = h, c
                x_t = h

        return h_t[-1]  # [B, hidden_dim, H, W]


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_hat):
        # x: [B, N, F], A_hat: [N, N]
        x = torch.matmul(A_hat, x)
        x = self.linear(x)
        x = F.relu(x)
        return x


class ExtremeEventForecaster(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_gcn_layers,
        gcn_hidden_dim,
        num_regions,
    ):
        super().__init__()
        self.encoder = ConvLSTMEncoder(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            kernel_size=3,
            num_layers=1,
        )
        self.num_regions = num_regions

        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        gcn_layers = []
        in_dim = hidden_dim
        for i in range(num_gcn_layers):
            out_dim = gcn_hidden_dim if i < num_gcn_layers - 1 else 1
            gcn_layers.append(GCNLayer(in_dim, out_dim))
            in_dim = out_dim
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self, x, A_hat):
        # x: [B, T, C, H, W]
        h = self.encoder(x)  # [B, hidden_dim, H, W]
        B, D, H, W = h.size()

        global_logits = self.global_head(h)  # [B, 1]

        h_flat = h.view(B, D, H * W).permute(0, 2, 1)  # [B, N, D]
        if H * W != self.num_regions:
            raise ValueError("num_regions must equal H*W in this implementation.")

        g = h_flat
        for layer in self.gcn_layers:
            g = layer(g, A_hat)
        regional_logits = g.squeeze(-1)  # [B, N]

        return global_logits, regional_logits
