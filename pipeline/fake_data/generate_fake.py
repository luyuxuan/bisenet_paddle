import sys
import paddle   
sys.path.append('.')
import numpy as np
def generate_fake_data(save_path):
    np.random.seed(33)
    fake_data = np.random.random(size=(2,3,1024,1024)).astype(np.float32) 
    np.save(save_path,fake_data)
def generate_fake_label(save_path):
    np.random.seed(33)
    fake_data = np.random.random(size=(2,1024,1024)).astype(np.int64) 
    np.save(save_path,fake_data)
# generate_fake_data("./fake_input_data.npy")
# generate_fake_label("./fake_input_label.npy")