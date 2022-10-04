import torch
from torch import nn
import torch.nn.functional as F

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=103, out_channels=32, kernel_size=3, stride=1, padding=0),  # output len=8
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            #nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),

            nn.Flatten()
        )
        self.fc11 = nn.Sequential(
            nn.Linear(in_features=8 * 16, out_features=100),
            nn.Linear(in_features=100, out_features=200),
            #nn.Linear(in_features=50, out_features=50)
        )
        self.fc12 = nn.Sequential(
            nn.Linear(in_features=8 * 16, out_features=100),
            nn.Linear(in_features=100, out_features=200),
            #nn.Linear(in_features=50, out_features=50)
        )
        self.linear = nn.Linear(in_features=200, out_features=103 * 10)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=10, hidden_size=10, num_layers=10)
    # 通过mu,sigma,误差 生成z
    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar * 0.5)
        eps = torch.randn_like(sigma)
        eps = eps.cuda()
        z = mu + sigma * eps
        return z
    def decoder(self, z):
        out3 = self.relu(self.linear(z))
        out3 = torch.reshape(out3, [103, 10])
        out3, _ = self.gru(out3)
        # 归一化结果
        out3 = F.softmax(out3, dim=0)
        out3 = torch.reshape(out3, [1, 103, 10])
        return out3
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out1 = self.encoder(x)
        out2 = self.encoder(x)
        mu = self.fc11(out1)
        logvar = self.fc12(out2)
        z = self.reparameterize(mu, logvar)
        # decode 部分，out3=GRU(GRU(GRU(MLP(z)))
        out3 = self.relu(self.linear(z))
        out3 = torch.reshape(out3, [103, 10])
        out3, _ = self.gru(out3)
        # 归一化结果
        out3 = F.softmax(out3, dim=0)
        out3 = torch.reshape(out3, [1, 103, 10])
        return out3, mu, logvar
