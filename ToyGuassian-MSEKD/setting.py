import numpy as np

K_CLAS = 10                          # Number of classes in the toy dataset
N_Data = int(1e5)                   # Number of samples in the dataset
TVT_SPLIT = [0.05, 0.05, 0.9]       # Split ratio between of train/valid/test dataset
N_Train = int(N_Data*TVT_SPLIT[0])  # Number of training samples
N_Valid = int(N_Data*TVT_SPLIT[1])  # Number of validation samples
N_Test = int(N_Data*TVT_SPLIT[2])   # Number of test samples

BATCH_SIZE = 32                     # Training batch size
EPOCHS = 100                         # Number of training epochs
LR = 5e-4                           # Initial learning rate
LR_MIN = 5e-4                       # Minimum learning rate in cosine scheduler
X_DIM = 30                          # Dimension of input signal x
NOISE = 6                          # Noisy level when generating the dataset

LS_EPS = (1-0.05)                          # Eps of label smoothing, label is y*eps + u*(1-eps)/K_CLAS

DELTA_MU = 1
MU_VEC = np.random.randint(-1,2,size=(K_CLAS,X_DIM)) * DELTA_MU     # mu_1,...,mu_K
MU_VEC_ALL = np.tile(MU_VEC,(N_Data,1,1))

SAVEDATA = False
USELOCALDATA = True
PATH = './data/K-10'

