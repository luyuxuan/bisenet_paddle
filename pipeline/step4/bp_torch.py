import sys
import paddle   
sys.path.append('.')
import numpy as np 
import torch
from for_torch.models.bisenetv1 import BiSeNetV1
from for_torch.lib.ohem_ce_loss import OhemCELoss
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

cfg = set_cfg_from_file("for_torch/lib/bisenetv1_city.py")


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        # print(nowd_params)
        params_list = [
            {'params': wd_params, },
            # {'params': nowd_params, 'weight_decay': 0},
            # {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            # {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        # params_list,
        model.parameters(),
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay
        # weight_decay=5e-4
    )
    # lr = optim.lr_get()
    
    return optim





if __name__ == "__main__":
    loss_torch_data = ReprodLogger()
    net = BiSeNetV1(19)
    net.load_state_dict(torch.load('weights/model_final_v1_city_new.pth', map_location='cpu'))
    net.eval()
    loss = OhemCELoss(0.7)
    optim = set_optimizer(net)

    for i in range(5):  
        fake_data = np.load("fake_data/fake_input_data.npy")
        fake_data = torch.from_numpy(fake_data)
        fake_label = np.load("fake_data/fake_input_label.npy")
        fake_label = torch.from_numpy(fake_label)
        print(fake_data.shape)
        optim.zero_grad()
        # print(lr)
        logits,logits2,logits3 = net(fake_data)
        # print(logits)
        loss_pre = loss(logits, fake_label)
        loss_pre2 = loss(logits2, fake_label)
        loss_pre3 = loss(logits3, fake_label)
        loss_tol = loss_pre + loss_pre2 + loss_pre3
        loss_tol.backward()
        # for name, tensor in net.named_parameters():
        #     grad = tensor.grad
        #     print(name, tensor.grad.shape)
            # break
        optim.step()
        # torch.cuda.synchronize()
        # lr_schdr.step()
        # print(loss_tol)
        loss_pre = loss_pre.detach().numpy() 
        loss_tol = loss_tol.detach().numpy() 
        print(loss_pre)
        print(loss_tol)



