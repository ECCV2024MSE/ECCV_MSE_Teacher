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

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

# seed_torch(1)

def sigmoid(x):
  return 1/(1+np.exp(-x))

def y_to_oht(label):
  label_oht = torch.zeros(label.shape[0],K_CLAS).to(label.device)
  label_oht.scatter_(1,label,1)
  label_oht = label_oht.float()
  return label_oht


def data_gen(x,y,p):
  '''
    Pack everything to a train_loader and a val_lodaer
  '''
  x, y, p = torch.tensor(x), torch.tensor(y), torch.tensor(p)
  dataset=Data.TensorDataset(x, y, p)
  indices = list(range(N_Data))
  np.random.shuffle(indices)
  train_indices, val_indices, test_indices = indices[:N_Train], indices[N_Train:N_Train+N_Valid], indices[N_Train+N_Valid:]
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices) 
  test_sampler = SubsetRandomSampler(test_indices) 
  train_loader = Data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last = True)
  valid_loader = Data.DataLoader(dataset, batch_size=N_Valid, sampler=valid_sampler, drop_last = True)
  test_loader = Data.DataLoader(dataset, batch_size=N_Test, sampler=test_sampler, drop_last = True)
  return train_loader, valid_loader, test_loader




def cal_ECE(pb_table, tf_table):
  '''
    pb_table is the probability provided by network
    tf_table is the acc results of the prodiction
  '''
  BM_acc = np.zeros((K_CLAS,))
  BM_conf = np.zeros((K_CLAS,))
  BM_cnt = np.zeros((K_CLAS,))
  Index_table = (pb_table.T*K_CLAS-1e-6).int().squeeze()

  for i in range(pb_table.shape[0]):
    idx = Index_table[i]
    BM_cnt[idx] += 1
    BM_conf[idx] += pb_table[i]
    if tf_table[i]:
      BM_acc[idx] += 1
  ECE = 0
  for j in range(K_CLAS):
    if BM_cnt[j] != 0:
      ECE += BM_cnt[j]*np.abs(BM_acc[j]/BM_cnt[j]-BM_conf[j]/BM_cnt[j])
  return ECE/BM_cnt.sum()

def L2_distance_logits_p(logits, p):
  q = F.softmax(logits,1)
  return L2_distance_q_p(q,p)
  
def L2_distance_q_p(q, p):
  return  torch.dist(q.reshape(-1,1),p.reshape(-1,1),p=2)#(nn.MSELoss(reduction='mean')(q.reshape(-1,1),p.reshape(-1,1)))**2

def CE_distance_q_p(q, p):
  # any prob
  # p is the true probability (e.g. true probability)
  # q = F.softmax(q, 1)
  # loss = nn.KLDivLoss(reduction = 'batchmean')
  # breakpoint()
  # return loss(torch.nan_to_num(q.log(), neginf = -10), p)
  out = -p * torch.clamp(q.log(), min = -10)
  return out.sum(1).mean()

def cal_entropy(logits, p):
  # logits is the output of the network
  # p is the true probability (e.g. true probability)
  return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits,1), p)

def _y_to_oht(label):
  label_oht = torch.zeros(label.shape[0],K_CLAS).to(label.device)
  label_oht.scatter_(1,label,1)
  label_oht = label_oht.float()
  return label_oht

def _y_to_smoothed(y):
  y_oht = _y_to_oht(y)
  return y_oht*LS_EPS + torch.ones_like(y_oht)*(1-LS_EPS)/K_CLAS

class MLP(nn.Module):
  def __init__(self, in_dim, hid_size=128):
    super(MLP, self).__init__()
    self.in_dim = in_dim
    self.hid_size = hid_size    
    self.fc1 = nn.Linear(self.in_dim, self.hid_size)
    self.fc2 = nn.Linear(self.hid_size, self.hid_size)
    self.fc3 = nn.Linear(self.hid_size, K_CLAS)
    self.act = nn.ReLU(True)

  def forward(self, x):
    h1 = self.act(self.fc1(x))
    h2 = self.act(self.fc2(h1))
    out = self.fc3(h2)
    return out
  
# TODO:
def eval_model_on_test(model, metric = L2_distance_logits_p):
  model.eval()
  for x,y,p in test_loader:
    x,y,p = x.float().cuda(), y.long(), p.float()
    break
  hid = model(x)
  hid = hid.cpu().detach()
  pred = hid.data.max(1, keepdim=True)[1]
  prob_table = torch.gather(nn.Softmax(1)(hid),dim=1,index=pred)
  tf_table = pred.eq(y.data.view_as(pred))
  acc = tf_table.sum()/N_Test
  dist = metric(hid, p)
  ECE = cal_ECE(prob_table,tf_table)
  model.train()
  return acc, dist, ECE



def get_validation(model, data_loader, loss_type='from_oht', teacher=None):
  batch_size = data_loader.batch_size
  model.eval()
  correct = 0
  dist_p,dist_tgt = 0, 0
  pb_table, tf_table = [], []
  hid_all, p_all, p_tgt_all = [], [], []
  b_cnt = 0
  for x, y, p in data_loader:
    b_cnt += 1
    x,y,p = x.float().cuda(), y.long(), p.float()
    with torch.no_grad():
      hid = model(x)
      hid = hid.cpu().detach()
      pred = hid.data.max(1, keepdim=True)[1] # get the index of the max log-probability
      prob = torch.gather(nn.Softmax(1)(hid),dim=1,index=pred)
      y_oht=y_to_oht(y.long())
      pb_table.append(prob)
      tf_table.append(pred.eq(y.data.view_as(pred)))     

      if loss_type == 'from_oht':
        p_tgt = _y_to_oht(y)
      elif loss_type == 'from_ls':
        p_tgt = _y_to_smoothed(y)
      elif loss_type == 'from_gt' or loss_type == 'noise_prob':
        p_tgt = p
      elif loss_type == 'from_teacher':              
        teacher.eval()
        hid_teach = teacher(x)
        hid_teach = hid_teach.cpu().detach()
        p_tgt = F.softmax(hid_teach,1)
      p_all.append(p)
      hid_all.append(hid)
      p_tgt_all.append(p_tgt)
  model.train()
  pb_table = torch.stack(pb_table).reshape(-1,1)
  tf_table = torch.stack(tf_table).reshape(-1,1)
  ECE = cal_ECE(pb_table, tf_table)
  B_NUM = batch_size*b_cnt
  correct = tf_table.sum()

  hid_all = torch.stack(hid_all).reshape(-1,K_CLAS)
  p_all = torch.stack(p_all).reshape(-1,K_CLAS)
  p_tgt_all = torch.stack(p_tgt_all).reshape(-1,K_CLAS)
  dist_p = L2_distance_logits_p(hid_all, p_all)
  dist_tgt = L2_distance_logits_p(hid_all, p_tgt_all)
  return correct/B_NUM, dist_p, dist_tgt, ECE

# =========== Generate all x, y and p===============================
y_true = np.random.randint(0,K_CLAS,[N_Data,1]).astype(np.float32)
mu_true = np.zeros((N_Data, X_DIM))
for i in range(N_Data):
  mu_true[i,:] = MU_VEC[y_true[i].astype(np.int64),:]
x_true = mu_true + np.random.randn(N_Data, X_DIM)*np.sqrt(NOISE)

logits = np.zeros((N_Data,K_CLAS))
for k in range(K_CLAS):
  logits[:,k] = np.linalg.norm(x_true - MU_VEC_ALL[:,k,:], axis=1)**2*(-0.5/NOISE)
logits = torch.from_numpy(logits)
p_true = F.softmax(logits, 1)

if SAVEDATA:
  save_data(PATH=PATH, x_true=x_true,y_true=y_true,p_true=p_true)
  print("Save Data", flush=True)
if USELOCALDATA:
  x_true,y_true,p_true = load_data(PATH=PATH)
  print("Load Data", flush=True)
train_loader, valid_loader, test_loader = data_gen(x_true,y_true,p_true)



def train(model, optimizer, scheduler, loss_type='from_oht', teacher=None, prt_flag=False, p_noise=0, noisy_type='combine',temp=1, criterion = 'CE'):
  results = {'loss':[], 'tacc':[], 'vacc':[], 'tdistp':[],'vdistp':[],'tdisttgt':[],'vdisttgt':[],'tECE':[],'vECE':[],'L2_ptgt_pgt':[],'IDX_MAX':[], 'CE':[], 'CE_reverse':[]}
  vacc_max = 0
  model.train()
  p_list, p_tgt_list = [], []
  for g in range(EPOCHS):
    for x,y,p in train_loader:
      x,y,p = x.float().cuda(), y.long().cuda(), p.float().cuda()
      # breakpoint()
      optimizer.zero_grad()
      hid = model(x)

      if loss_type == 'from_oht':
        p_tgt = _y_to_oht(y)
      elif loss_type == 'from_ls':
        p_tgt = _y_to_smoothed(y)
      elif loss_type == 'from_gt':
        p_tgt = p
      elif loss_type == 'from_teacher':
        teacher.eval()
        hid_teach = teacher(x)
        hid_teach = hid_teach.detach()
        p_tgt = F.softmax(hid_teach/temp,1)
      elif loss_type == 'noise_prob':
        if noisy_type=='combine':
          p_tgt = _y_to_oht(y)*p_noise[0] + p*(1-p_noise[0])
        elif noisy_type=='plus':
          noisy_p = p + torch.from_numpy(np.random.randn(p.shape[0],p.shape[1])*np.sqrt(p_noise)).cuda()
          noisy_p = torch.clamp(noisy_p,min=1e-6,max=2)
          noisy_p = noisy_p/noisy_p.sum(1).expand([K_CLAS,-1]).transpose(0,1)
          p_tgt = noisy_p.float()          
      if g==0:          # Only calculate L2_ptgt_pgt once
        p_list.append(p)
        p_tgt_list.append(p_tgt)
      if criterion.upper() == 'MSE':
        loss = F.mse_loss(F.softmax(hid, 1), p_tgt)
      # elif loss_type == 'from_teacher':
      #   # breakpoint()
      #   loss = .9 * cal_entropy(hid/temp, p_tgt)/(temp**2) + .1 * F.cross_entropy(hid, y.flatten())
      else:
        loss = cal_entropy(hid/temp, p_tgt)/(temp**2)
      loss.backward()
      optimizer.step()
      results['loss'].append(loss.item())
    # ---------- At the end of each epoch ----------
    tacc, tdistp, tdisttgt, tECE = get_validation(model, data_loader=train_loader, loss_type=loss_type, teacher=teacher)
    vacc, vdistp, vdisttgt, vECE = get_validation(model, data_loader=valid_loader, loss_type=loss_type, teacher=teacher)
    results['tacc'].append(tacc)
    results['vacc'].append(vacc)
    results['tdistp'].append(tdistp)
    results['vdistp'].append(vdistp)
    results['tdisttgt'].append(tdisttgt)
    results['vdisttgt'].append(vdisttgt)
    results['tECE'].append(tECE)
    results['vECE'].append(vECE)

    if vacc>vacc_max:
      vacc_max = vacc
      ES_model = copy.deepcopy(model)
      results['IDX_MAX']=g
    if prt_flag and g%10==0:
      print('\t==Training , Epoch: {:3d}/{:3d}\tLoss: {:.6f}\tTACC: {:.6f},\tVACC:{:.6f}'.format(g,EPOCHS, results['loss'][-1], tacc, vacc))
    scheduler.step()
  results['L2_ptgt_pgt'] = L2_distance_q_p(torch.stack(p_list).reshape(-1,K_CLAS), torch.stack(p_tgt_list).reshape(-1,K_CLAS)).cpu()
  results['CE_reverse'] = CE_distance_q_p(torch.stack(p_list).reshape(-1,K_CLAS), torch.stack(p_tgt_list).reshape(-1,K_CLAS)).cpu()
  results['CE'] = CE_distance_q_p(torch.stack(p_tgt_list).reshape(-1,K_CLAS), torch.stack(p_list).reshape(-1,K_CLAS)).cpu()
  return ES_model, results

def show_test_results(model, type_='OHT'):
  acc, distp, ECE = eval_model_on_test(model)
  print(type_+': \t Test acc is %.5f; tdistp is %.5f; ECE is %.5f'%(acc.item(), distp.item(), ECE.item()))
  return acc, distp, ECE

dist_kd_to_p_mse = -1 * np.ones(10)
ce_kd_to_p_mse = -1 * np.ones(10)
# dist_eskd_to_p_mse = -1 * np.ones(10)
KD_test_acc_mse = -1 * np.ones(10)
# ESKD_test_acc_mse = -1 * np.ones(10)
criterion = 'MSE'

for i in tqdm(range(10)):
  print("Train One Hot Model")
  OHT_model = MLP(in_dim=X_DIM).cuda()
  OHT_optimizer = optim.SGD(OHT_model.parameters(), lr=LR, momentum=0.9)
  OHT_scheduler = optim.lr_scheduler.CosineAnnealingLR(OHT_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  best_OHT_model, OHT_results = train(OHT_model, OHT_optimizer, OHT_scheduler, 'from_oht',prt_flag=True, criterion=criterion)

  # print("Train Label Smoothing Model")
  # LS_EPS = (1-0.03)   
  # LS_model = MLP(in_dim=X_DIM).cuda()
  # LS_optimizer = optim.SGD(LS_model.parameters(), lr=LR, momentum=0.9)
  # LS_scheduler = optim.lr_scheduler.CosineAnnealingLR(LS_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  # best_LS_model, LS_results = train(LS_model, LS_optimizer, LS_scheduler, 'from_ls',prt_flag=True)

  # print("Train Ground Truth Model")
  # GT_model = MLP(in_dim=X_DIM).cuda()
  # GT_optimizer = optim.SGD(GT_model.parameters(), lr=LR, momentum=0.9)
  # GT_scheduler = optim.lr_scheduler.CosineAnnealingLR(GT_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  # best_GT_model, GT_results = train(GT_model, GT_optimizer, GT_scheduler, 'from_gt',prt_flag=True)

  print("Train KD Model")
  KD_model = MLP(in_dim=X_DIM).cuda()
  KD_optimizer = optim.SGD(KD_model.parameters(), lr=LR, momentum=0.9)
  KD_scheduler = optim.lr_scheduler.CosineAnnealingLR(KD_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  best_KD_model, KD_results = train(KD_model, KD_optimizer, KD_scheduler, 'from_teacher',teacher=OHT_model,prt_flag=True)

  # print("Train Earily Stop KD Model")
  # ESKD_model = MLP(in_dim=X_DIM).cuda()
  # ESKD_optimizer = optim.SGD(ESKD_model.parameters(), lr=LR, momentum=0.9)
  # ESKD_scheduler = optim.lr_scheduler.CosineAnnealingLR(ESKD_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  # best_ESKD_model, ESKD_results = train(ESKD_model, ESKD_optimizer, ESKD_scheduler, 'from_teacher',teacher=best_OHT_model,prt_flag=True)

  tmp_oht_q, tmp_ls_q, tmp_kd_q, tmp_eskd_q, tmp_all_p = [], [], [], [], []

  for tx,ty,tp in train_loader:
    tx,ty = tx.float().cuda(),ty.long()
    kd_hid = OHT_model(tx).cpu()
    kd_pred = nn.Softmax(1)(kd_hid)
    eskd_hid = best_OHT_model(tx).cpu()
    eskd_pred = nn.Softmax(1)(eskd_hid)
    ls = (_y_to_oht(ty)*LS_EPS+(1-LS_EPS)/(K_CLAS-1)*torch.ones(BATCH_SIZE,K_CLAS))

    tmp_oht_q.append(_y_to_oht(ty))
    tmp_ls_q.append(ls)
    tmp_kd_q.append(kd_pred)
    tmp_eskd_q.append(eskd_pred)
    tmp_all_p.append(tp)
  tmp_oht_q = torch.stack(tmp_oht_q).reshape(-1,K_CLAS)
  tmp_ls_q = torch.stack(tmp_ls_q).reshape(-1,K_CLAS)
  tmp_kd_q = torch.stack(tmp_kd_q).reshape(-1,K_CLAS)
  tmp_eskd_q = torch.stack(tmp_eskd_q).reshape(-1,K_CLAS)
  tmp_all_p = torch.stack(tmp_all_p).reshape(-1,K_CLAS)

  dist_oht_to_p = L2_distance_q_p(tmp_oht_q, tmp_all_p).item()
  dist_ls_to_p = L2_distance_q_p(tmp_ls_q, tmp_all_p).item()
  dist_kd_to_p = L2_distance_q_p(tmp_kd_q, tmp_all_p).item()
  dist_eskd_to_p = L2_distance_q_p(tmp_eskd_q, tmp_all_p).item()
  ce_oht_to_p = CE_distance_q_p(tmp_oht_q, tmp_all_p).item()
  ce_ls_to_p = CE_distance_q_p(tmp_ls_q, tmp_all_p).item()
  ce_kd_to_p = CE_distance_q_p(tmp_kd_q, tmp_all_p).item()
  ce_eskd_to_p = CE_distance_q_p(tmp_eskd_q, tmp_all_p).item()
  ce_oht_to_p_reverse = CE_distance_q_p(tmp_all_p, tmp_oht_q).item()
  ce_ls_to_p_reverse = CE_distance_q_p(tmp_all_p, tmp_ls_q).item()
  ce_kd_to_p_reverse = CE_distance_q_p(tmp_all_p, tmp_kd_q).item()
  ce_eskd_to_p_reverse = CE_distance_q_p(tmp_all_p, tmp_eskd_q).item()
  # print('\nL2 DISTANCE:')
  # print('OHT\t'+str(dist_oht_to_p))
  # print('LS\t'+str(dist_ls_to_p))
  # print('KD\t'+str(dist_kd_to_p))
  # print('ESKD\t'+str(dist_eskd_to_p))
  # print('\nCE DISTANCE:')
  # print('OHT\t'+str(ce_oht_to_p))
  # print('LS\t'+str(ce_ls_to_p))
  # print('KD\t'+str(ce_kd_to_p))
  # print('ESKD\t'+str(ce_eskd_to_p))
  # print('\nCE DISTANCE REVERSE:')
  # print('OHT\t'+str(ce_oht_to_p_reverse))
  # print('LS\t'+str(ce_ls_to_p_reverse))
  # print('KD\t'+str(ce_kd_to_p_reverse))
  # print('ESKD\t'+str(ce_eskd_to_p_reverse))
  # print()
  KD_test_acc, KD_test_distp, KD_test_ECE = show_test_results(best_KD_model,' KD ')
  # ESKD_test_acc, ESKD_test_distp, ESKD_test_ECE = show_test_results(best_ESKD_model,'ESKD')
  dist_kd_to_p_mse[i] = dist_kd_to_p
  # dist_eskd_to_p_mse[i] = dist_eskd_to_p
  ce_kd_to_p_mse[i] = ce_kd_to_p_reverse
  KD_test_acc_mse[i] = KD_test_acc
  # ESKD_test_acc_mse[i] = ESKD_test_acc

dist_kd_to_p_ce = -1 * np.ones(10)
ce_kd_to_p_ce = -1 * np.ones(10)
# dist_eskd_to_p_ce = -1 * np.ones(10)
KD_test_acc_ce = -1 * np.ones(10)
# ESKD_test_acc_ce = -1 * np.ones(10)
criterion = 'CE'

for i in tqdm(range(10)):
  print("Train One Hot Model")
  OHT_model = MLP(in_dim=X_DIM).cuda()
  OHT_optimizer = optim.SGD(OHT_model.parameters(), lr=LR, momentum=0.9)
  OHT_scheduler = optim.lr_scheduler.CosineAnnealingLR(OHT_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  best_OHT_model, OHT_results = train(OHT_model, OHT_optimizer, OHT_scheduler, 'from_oht',prt_flag=True, criterion=criterion)

  # print("Train Label Smoothing Model")
  # LS_EPS = (1-0.03)   
  # LS_model = MLP(in_dim=X_DIM).cuda()
  # LS_optimizer = optim.SGD(LS_model.parameters(), lr=LR, momentum=0.9)
  # LS_scheduler = optim.lr_scheduler.CosineAnnealingLR(LS_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  # best_LS_model, LS_results = train(LS_model, LS_optimizer, LS_scheduler, 'from_ls',prt_flag=True)

  # print("Train Ground Truth Model")
  # GT_model = MLP(in_dim=X_DIM).cuda()
  # GT_optimizer = optim.SGD(GT_model.parameters(), lr=LR, momentum=0.9)
  # GT_scheduler = optim.lr_scheduler.CosineAnnealingLR(GT_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  # best_GT_model, GT_results = train(GT_model, GT_optimizer, GT_scheduler, 'from_gt',prt_flag=True)

  print("Train KD Model")
  KD_model = MLP(in_dim=X_DIM).cuda()
  KD_optimizer = optim.SGD(KD_model.parameters(), lr=LR, momentum=0.9)
  KD_scheduler = optim.lr_scheduler.CosineAnnealingLR(KD_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  best_KD_model, KD_results = train(KD_model, KD_optimizer, KD_scheduler, 'from_teacher',teacher=OHT_model,prt_flag=True)

  # print("Train Earily Stop KD Model")
  # ESKD_model = MLP(in_dim=X_DIM).cuda()
  # ESKD_optimizer = optim.SGD(ESKD_model.parameters(), lr=LR, momentum=0.9)
  # ESKD_scheduler = optim.lr_scheduler.CosineAnnealingLR(ESKD_optimizer, T_max=EPOCHS, eta_min=LR_MIN)
  # best_ESKD_model, ESKD_results = train(ESKD_model, ESKD_optimizer, ESKD_scheduler, 'from_teacher',teacher=best_OHT_model,prt_flag=True)

  tmp_oht_q, tmp_ls_q, tmp_kd_q, tmp_eskd_q, tmp_all_p = [], [], [], [], []

  for tx,ty,tp in train_loader:
    tx,ty = tx.float().cuda(),ty.long()
    kd_hid = OHT_model(tx).cpu()
    kd_pred = nn.Softmax(1)(kd_hid)
    eskd_hid = best_OHT_model(tx).cpu()
    eskd_pred = nn.Softmax(1)(eskd_hid)
    ls = (_y_to_oht(ty)*LS_EPS+(1-LS_EPS)/(K_CLAS-1)*torch.ones(BATCH_SIZE,K_CLAS))

    tmp_oht_q.append(_y_to_oht(ty))
    tmp_ls_q.append(ls)
    tmp_kd_q.append(kd_pred)
    tmp_eskd_q.append(eskd_pred)
    tmp_all_p.append(tp)
  tmp_oht_q = torch.stack(tmp_oht_q).reshape(-1,K_CLAS)
  tmp_ls_q = torch.stack(tmp_ls_q).reshape(-1,K_CLAS)
  tmp_kd_q = torch.stack(tmp_kd_q).reshape(-1,K_CLAS)
  tmp_eskd_q = torch.stack(tmp_eskd_q).reshape(-1,K_CLAS)
  tmp_all_p = torch.stack(tmp_all_p).reshape(-1,K_CLAS)

  dist_oht_to_p = L2_distance_q_p(tmp_oht_q, tmp_all_p).item()
  dist_ls_to_p = L2_distance_q_p(tmp_ls_q, tmp_all_p).item()
  dist_kd_to_p = L2_distance_q_p(tmp_kd_q, tmp_all_p).item()
  dist_eskd_to_p = L2_distance_q_p(tmp_eskd_q, tmp_all_p).item()
  ce_oht_to_p = CE_distance_q_p(tmp_oht_q, tmp_all_p).item()
  ce_ls_to_p = CE_distance_q_p(tmp_ls_q, tmp_all_p).item()
  ce_kd_to_p = CE_distance_q_p(tmp_kd_q, tmp_all_p).item()
  ce_eskd_to_p = CE_distance_q_p(tmp_eskd_q, tmp_all_p).item()
  ce_oht_to_p_reverse = CE_distance_q_p(tmp_all_p, tmp_oht_q).item()
  ce_ls_to_p_reverse = CE_distance_q_p(tmp_all_p, tmp_ls_q).item()
  ce_kd_to_p_reverse = CE_distance_q_p(tmp_all_p, tmp_kd_q).item()
  ce_eskd_to_p_reverse = CE_distance_q_p(tmp_all_p, tmp_eskd_q).item()
  # print('\nL2 DISTANCE:')
  # print('OHT\t'+str(dist_oht_to_p))
  # print('LS\t'+str(dist_ls_to_p))
  # print('KD\t'+str(dist_kd_to_p))
  # print('ESKD\t'+str(dist_eskd_to_p))
  # print('\nCE DISTANCE:')
  # print('OHT\t'+str(ce_oht_to_p))
  # print('LS\t'+str(ce_ls_to_p))
  # print('KD\t'+str(ce_kd_to_p))
  # print('ESKD\t'+str(ce_eskd_to_p))
  # print('\nCE DISTANCE REVERSE:')
  # print('OHT\t'+str(ce_oht_to_p_reverse))
  # print('LS\t'+str(ce_ls_to_p_reverse))
  # print('KD\t'+str(ce_kd_to_p_reverse))
  # print('ESKD\t'+str(ce_eskd_to_p_reverse))
  # print()
  KD_test_acc, KD_test_distp, KD_test_ECE = show_test_results(best_KD_model,' KD ')
  # ESKD_test_acc, ESKD_test_distp, ESKD_test_ECE = show_test_results(best_ESKD_model,'ESKD')
  dist_kd_to_p_ce[i] = dist_kd_to_p
  ce_kd_to_p_ce[i] = ce_kd_to_p_reverse
  # dist_eskd_to_p_ce[i] = dist_eskd_to_p
  KD_test_acc_ce[i] = KD_test_acc
  # ESKD_test_acc_ce[i] = ESKD_test_acc

np.save('./Project1/results/MSEKD_L2_dist_N_{}_K_{}_reverse.npy'.format(NOISE, K_CLAS), dist_kd_to_p_mse)
np.save('./Project1/results/MSEKD_ENT_dist_N_{}_K_{}_reverse.npy'.format(NOISE, K_CLAS), ce_kd_to_p_mse)
np.save('./Project1/results/MSEKD_test_acc_N_{}_K_{}_reverse.npy'.format(NOISE, K_CLAS), KD_test_acc_mse)
np.save('./Project1/results/CEKD_L2_dist_N_{}_K_{}_reverse.npy'.format(NOISE, K_CLAS), dist_kd_to_p_ce)
np.save('./Project1/results/CEKD_ENT_dist_N_{}_K_{}_reverse.npy'.format(NOISE, K_CLAS), ce_kd_to_p_ce)
np.save('./Project1/results/CEKD_test_acc_N_{}_K_{}_reverse.npy'.format(NOISE, K_CLAS), KD_test_acc_ce)

OHT_test_acc, OHT_test_distp, OHT_test_ECE = show_test_results(best_OHT_model, 'OHT ')
# LS_test_acc, LS_test_distp, LS_test_ECE = show_test_results(best_LS_model, ' LS ')
# GT_test_acc, GT_test_distp, GT_test_ECE = show_test_results(best_GT_model, ' GT ')
KD_test_acc, KD_test_distp, KD_test_ECE = show_test_results(best_KD_model,' KD ')
# ESKD_test_acc, ESKD_test_distp, ESKD_test_ECE = show_test_results(best_ESKD_model,'ESKD')

fig = plt.figure()
ax1 = fig.add_subplot(121)

ax1.scatter(dist_kd_to_p_mse, KD_test_acc_mse, color = 'red', alpha = .6, label = 'KD-MSE')
# plt.scatter(dist_eskd_to_p_mse, ESKD_test_acc_mse, color = 'red', marker = '^', alpha = .6, label = 'ESKD-MSE')
ax1.scatter(dist_kd_to_p_ce, KD_test_acc_ce, color = 'blue', alpha = .6, label = 'KD-CE')
# plt.scatter(dist_eskd_to_p_ce, ESKD_test_acc_ce, color = 'blue', marker = '^', alpha = .6, label = 'ESKD-CE')

# plt.legend(fontsize = 12)
ax1.set_ylabel('Accuracy on test set', fontsize=16)
ax1.set_xlabel('MSE of p_tar and p*',fontsize=16)

ax2 = fig.add_subplot(122)

ax2.scatter(ce_kd_to_p_mse, KD_test_acc_mse, color = 'red', alpha = .6, label = 'MSE')
# plt.scatter(dist_eskd_to_p_mse, ESKD_test_acc_mse, color = 'red', marker = '^', alpha = .6, label = 'ESKD-MSE')
ax2.scatter(ce_kd_to_p_ce, KD_test_acc_ce, color = 'blue', alpha = .6, label = 'CE')
# plt.scatter(dist_eskd_to_p_ce, ESKD_test_acc_ce, color = 'blue', marker = '^', alpha = .6, label = 'ESKD-CE')

ax2.legend(fontsize = 12)
ax2.set_ylabel('Accuracy on test set', fontsize=16)
ax2.set_xlabel('CE of p_tar and p*',fontsize=16)


plt.show()
