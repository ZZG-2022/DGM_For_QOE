import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 该文件用于构建qovae
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from QoVAE_1 import Mynet

EPOCH = 100
log_interval = 10

# 加载数据集
class Mydata(Dataset):
    def __init__(self):
        self.dataset_file = np.load("./qovae_code/dataset_choosed_file.npy")
        self.dataset_file = torch.tensor(self.dataset_file, dtype=torch.float32)
        self.dataset_file = self.dataset_file[:, :, :]

    def __getitem__(self, index):  # index编号
        sequence_data = self.dataset_file[index, :, :]
        return sequence_data

    def __len__(self):
        return len(self.dataset_file[:, 1, 1])


# 构建网络
dataset = Mydata()
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, recon_x, x, mu, logvar, ratio):
        return loss_func(recon_x, x, mu, logvar, ratio)


def loss_func(recon_x, x, mu, logvar, ratio):
    BCE = F.binary_cross_entropy(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + ratio*KLD,BCE,KLD


qovae = Mynet().cuda()
#print(qovae)
# qovae = Mynet()
loss = Loss()
#qovae.load_state_dict(torch.load("qovae_model_init.pth"))
#optimizer = optim.Adam(qovae.parameters(), lr=0.0002, betas=(0.9, 0.999))
optimizer = optim.SGD(qovae.parameters(), lr=0.0001, momentum=0.9)
optimizer.zero_grad()

onehot_matrix = []
writer = SummaryWriter("./logs_seq_1")
LOSS_seq = []
LOSS_KLD_seq = []
LOSS_BCE_seq = []
for i in range(EPOCH):

    running_loss = 0.0
    KLD_loss = 0
    BCE_loss = 0
    if i < 20:
        ratio = np.exp(i/20-1)
    else:
        ratio = 1
    #ratio = abs(np.sin(i*np.pi/15))
    # print(torch.cuda_version)
    # print(torch.cuda.is_available())
    for data in train_loader:
        onehot_matrix = data
        onehot_matrix = onehot_matrix.cuda()
        output_x, mu, logvar = qovae(onehot_matrix)
        result_loss,KLD,BCE = loss(output_x, onehot_matrix.permute(0, 2, 1), mu, logvar,ratio)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        KLD_loss += KLD
        BCE_loss += BCE
        running_loss = running_loss + result_loss
    LOSS_seq.append(running_loss)
    LOSS_KLD_seq.append(KLD_loss)
    LOSS_BCE_seq.append(BCE_loss)
    writer.add_scalar(tag='loss_1', scalar_value=running_loss, global_step=i)
    if i % 2 == 0:
        print(running_loss)


#writer.add_graph(qovae, onehot_matrix)
writer.close()

#torch.save(qovae, "qovae_model_test.pth")
torch.save(qovae.state_dict(), "qovae_model_2.pth")

LOSS_seq = Tensor(LOSS_seq)
LOSS_seq = Tensor.cpu(LOSS_seq)
LOSS_BCE_seq = Tensor(LOSS_BCE_seq)
LOSS_BCE_seq = Tensor.cpu(LOSS_BCE_seq)
LOSS_KLD_seq = Tensor(LOSS_KLD_seq)
LOSS_KLD_seq = Tensor.cpu(LOSS_KLD_seq)

L = len(LOSS_seq)
X_label = [i for i in range(L)]
plt.plot(X_label, LOSS_seq, color='r')
plt.plot(X_label, LOSS_BCE_seq, color='g')
plt.plot(X_label, LOSS_KLD_seq, color='b')
plt.xlabel('loss')
plt.show()
