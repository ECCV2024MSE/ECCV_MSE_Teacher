import numpy as np
import os
def load_data(PATH):
    x_true = np.load(os.path.join(PATH, "x_true.npy"))
    y_true = np.load(os.path.join(PATH, "y_true.npy"))
    p_true = np.load(os.path.join(PATH, "p_true.npy"))
    return x_true,y_true,p_true

def save_data(PATH, x_true, y_true,p_true):
    np.save(os.path.join(PATH, "x_true.npy"), x_true)
    np.save(os.path.join(PATH, "y_true.npy"), y_true)
    np.save(os.path.join(PATH, "p_true.npy"), p_true)

