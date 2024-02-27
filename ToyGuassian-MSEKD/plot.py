import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import random
import copy
import scipy
import warnings
warnings.filterwarnings("ignore")

from setting import *
from utils import *



import numpy as np
from setting import *
import torch
import torch.utils.data as Data 
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from centroid import *


PATH = './Project1/results/'

L2_ptgt_pgt_list_combine = np.load(os.path.join(PATH, 'L2_ptgt_pgt_list_combine_N6K_3_DELTA_MU_4.npy'))
test_acc_list_combine = np.load(os.path.join(PATH, 'test_acc_list_combine_N6K_3_DELTA_MU_4.npy'))
CE_list_combine = np.load(os.path.join(PATH, 'CE_ptgt_pgt_list_combine_N6K_3_DELTA_MU_4.npy'))
CE_list_combine_reverse = np.load(os.path.join(PATH, 'CE_reverse_ptgt_pgt_list_combine_N6K_3_DELTA_MU_4.npy'))

L2_ptgt_pgt_list_combine = L2_ptgt_pgt_list_combine[CE_list_combine >= 0]
test_acc_list_combine = test_acc_list_combine[CE_list_combine >= 0]
CE_list_combine = CE_list_combine[CE_list_combine >= 0]


fig = plt.figure()
ax1 = fig.add_subplot(121)
# breakpoint()

ax1.scatter(L2_ptgt_pgt_list_combine, test_acc_list_combine, color='blue',alpha=0.6,label='Noisy p')
# ax1.scatter(dist_oht_to_p,OHT_test_acc,color='green',label='OHT',marker='2',s=400,linewidth=4)
# ax1.scatter(dist_ls_to_p,LS_test_acc,color='orange',label='LS',marker='2',s=400,linewidth=4)
# ax1.scatter(0,GT_test_acc,color='red',label='GT',marker='2',s=400,linewidth=4)
# ax1.scatter(dist_kd_to_p,KD_test_acc,color='cyan',label='KD',marker='2',s=400,linewidth=4)
# ax1.scatter(dist_eskd_to_p,ESKD_test_acc,color='purple',label='ESKD',marker='2',s=400,linewidth=4)

ax1.legend(fontsize=12)
ax1.set_ylabel('Accuracy on test set', fontsize=16)
ax1.set_xlabel('MSE of p_tar and p*',fontsize=16)

ax2 = fig.add_subplot(122)
# breakpoint()
ax2.scatter(CE_list_combine, test_acc_list_combine, color='blue',alpha=0.6,label='Noisy p')
# ax2.scatter(ce_oht_to_p,OHT_test_acc,color='green',label='OHT',marker='2',s=400,linewidth=4)
# ax2.scatter(ce_ls_to_p,LS_test_acc,color='orange',label='LS',marker='2',s=400,linewidth=4)
# ax2.scatter(0,GT_test_acc,color='red',label='GT',marker='2',s=400,linewidth=4)
# ax2.scatter(ce_kd_to_p,KD_test_acc,color='cyan',label='KD',marker='2',s=400,linewidth=4)
# ax2.scatter(ce_eskd_to_p,ESKD_test_acc,color='purple',label='ESKD',marker='2',s=400,linewidth=4)

ax2.legend(fontsize=12)
ax2.set_ylabel('Accuracy on test set', fontsize=16)
ax2.set_xlabel('CE of p_tar and p*',fontsize=16)

# ax3 = fig.add_subplot(133)

# ax3.scatter(CE_list_combine_reverse, test_acc_list_combine, color='blue',alpha=0.6,label='Noisy p')
# ax3.legend(fontsize=12)
# ax3.set_ylabel('Accuracy on test set', fontsize=16)
# ax3.set_xlabel('CE of p* and p_tar',fontsize=16)

ax2.set_title('K = {}, $\delta_\mu$ = {}'.format(K_CLAS, DELTA_MU), fontsize = 16)

plt.show()
print('MSE:', scipy.stats.spearmanr(L2_ptgt_pgt_list_combine, test_acc_list_combine))
print('CE:', scipy.stats.spearmanr(CE_list_combine, test_acc_list_combine))
