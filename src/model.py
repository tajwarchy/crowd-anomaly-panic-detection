import torch
import torch.nn as nn
import pickle


class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core':
            module = 'numpy.core'
        return super().find_class(module, name)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMClassifier(nn.Module):
    def __init__(self, in_ch, hidden_dim, num_classes, feat_h, feat_w):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.cell = ConvLSTMCell(in_ch, hidden_dim)
        pool_h = feat_h // 4
        pool_w = feat_w // 4
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_h, pool_w)),
            nn.Flatten(),
            nn.Linear(hidden_dim * pool_h * pool_w, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return self.classifier(h)


class CNN3DClassifier(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=(3,3,3), padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.classifier(self.encoder(x))


def build_model(cfg, weights_only_load=False):
    arch = cfg["training"]["architecture"]
    in_ch = cfg["training"]["feature_channels"]
    hidden_dim = cfg["training"]["hidden_dim"]
    num_classes = cfg["training"]["num_classes"]
    feat_h = cfg["sliding_window"]["feature_height"]
    feat_w = cfg["sliding_window"]["feature_width"]

    if arch == "convlstm":
        model = ConvLSTMClassifier(in_ch, hidden_dim, num_classes, feat_h, feat_w)
    else:
        model = CNN3DClassifier(in_ch, num_classes)
    return model


def load_model(cfg):
    model = build_model(cfg)
    ckpt_path = cfg["inference"]["checkpoint_path"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Model] Loaded epoch={ckpt['epoch']} val_auc={ckpt['val_auc']:.4f}")
    return model