import numpy as np
import torch
from numpy import shape
from QoVAE_1 import Mynet
import main_entropy_1 as M

devices = M.devices


# 将结果变成实验序列
def translation(L):
    L = L.cpu().detach().numpy()
    len_L = shape(L)
    sample = []
    a = len_L[0:1]
    for i in range(len_L[2]):
        Index_L = np.argmax(L[0, :, i])
        # 将L改为onehot类型
        L[0, :, i] = np.zeros(a)
        L[0, Index_L, i] = 1

        if Index_L != len_L[1] - 1:
            # 将L转换为装置序列
            sample.append(devices[Index_L])
    return sample


QoVAE = Mynet()
QoVAE.load_state_dict(torch.load("qovae_model_1.pth"))
# for name, param in QoVAE.named_parameters():
# print(name,':',param.size())

for i in range(10):
    pass
    Z = torch.randn([1, 200])
    L = QoVAE.decoder(Z)
    sample = translation(L)
    print(sample)
    S = M.comput_entropy(sample)
    if S != 0:
        print(S)
