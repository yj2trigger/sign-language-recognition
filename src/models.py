"""비교할 두 모델: 양방향 GRU 와 TCN(1D-CNN). 입력 (B, T, F) -> 로짓 (B, 클래스 수).
두 모델의 파라미터 수를 비슷하게(약 20만, 클래스 100개 기준) 맞춰 구조 차이만 비교한다."""
import torch
from torch import nn


class GRUNet(nn.Module):
    def __init__(self, n_feat, n_class, hidden=64, dropout=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(128, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_class))

    def forward(self, x):
        h, _ = self.rnn(self.inp(x))
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=1))


class _Block(nn.Module):
    def __init__(self, c, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation), nn.BatchNorm1d(c))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TCNNet(nn.Module):
    def __init__(self, n_feat, n_class, channels=80, dropout=0.3):
        super().__init__()
        self.inp = nn.Conv1d(n_feat, channels, 1)
        self.blocks = nn.Sequential(*[_Block(channels, d, dropout) for d in (1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * channels, n_class))

    def forward(self, x):
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.head(torch.cat([h.mean(2), h.amax(2)], dim=1))


class MLPBase(nn.Module):
    """기준선: 시간 순서를 버리고 프레임 특징의 평균·최대만 쓰는 MLP. GRU·TCN 이 이보다 얼마나 나은지가 '순서 모델링의 이득'이다."""

    def __init__(self, n_feat, n_class, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * n_feat, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_class))

    def forward(self, x):
        return self.net(torch.cat([x.mean(1), x.amax(1)], dim=1))


def build(name, n_feat, n_class):
    return {"gru": GRUNet, "tcn": TCNNet, "mlp": MLPBase}[name](n_feat, n_class)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
