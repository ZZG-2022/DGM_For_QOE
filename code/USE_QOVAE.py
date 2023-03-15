import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import shape

import main_entropy_1 as M
from QoVAE_1 import Mynet

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
QoVAE.load_state_dict(torch.load("qovae_model_2.pth"))
# for name, param in QoVAE.named_parameters():
# print(name,':',param.size())
k = 0
ZZ = [[],[]]
SS = []
for i in range(50):
    Z = torch.randn([1, 2])
    ZZ[0].append(Z[0][0])
    ZZ[1].append(Z[0][1])
    L = QoVAE.decoder(Z)
    sample = translation(L)
    S = abs(M.comput_entropy(sample))
    SS.append(S)
    print(sample)
    if S > 0:
        k += 1
        #print(sample)
        #print(S)

plt.scatter(x=ZZ[0],y=ZZ[1],c=SS,s=0.7)
plt.colorbar()
#print(ZZ)
print(SS)
print(k)
plt.show()