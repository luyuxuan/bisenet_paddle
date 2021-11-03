import sys
import paddle   
sys.path.append('.')
import numpy as np 
import torch
from for_torch.models.bisenetv1 import BiSeNetV1
from for_torch.lib.eval import miou
from for_torch.lib.get_dataloader import build_torch_data_pipeline
from reprod_log import ReprodLogger
import importlib
class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d
def set_cfg_from_file(cfg_path):
    spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec_loader = spec.loader.exec_module(cfg_file)
    cfg = cfg_file.cfg
    return cfg_dict(cfg)

cfg_torch = set_cfg_from_file("for_torch/lib/bisenetv1_city.py")

if __name__ == "__main__":
    miou_torch_data = ReprodLogger()
    net = BiSeNetV1(19)
    net.load_state_dict(torch.load('weights/model_final_v1_city_new.pth', map_location='cpu'))
    net.eval()
    torch_dataset, torch_dataloader = build_torch_data_pipeline(cfg_torch, mode='val')
#one img
    # fake_data = np.load("fake_data/fake_input_data.npy")
    # fake_data = torch.from_numpy(fake_data)
    # fake_label = np.load("fake_data/fake_input_label.npy")
    # fake_label = torch.from_numpy(fake_label)
    # print("--------fakedata",fake_data.shape)
    # print("--------fakelabel",fake_label.shape)
    # out = net(fake_data)[0]
    # print("-------->",out.shape,fake_label.shape)
    # miou_result = miou(out,fake_label,19)
    # miou_result = np.array(miou_result).astype(np.float32)
    # print(miou_result)
#dataset
    for i,(img,label) in enumerate(torch_dataloader):
        out = net(img)[0]
        N,_,H,W=label.shape
        label = label.view(N,H,W)
        miou_result = miou(out,label,19)
        miou_result = np.array(miou_result).astype(np.float32)
        print("miou_result:",miou_result)
        miou_torch_data.add(f"miou_{i}", np.array(miou_result))
    print("save the data")
    miou_torch_data.save("step2/metric_torch.npy")





