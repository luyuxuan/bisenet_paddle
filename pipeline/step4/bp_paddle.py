import sys
import paddle   
sys.path.append('.')
import numpy as np
from reprod_log import ReprodLogger
from for_paddle.models.bisenetv1 import BiSeNetV1
from for_paddle.lib.ohem_loss import OhemCrossEntropyLoss
from for_paddle.lib.lr_sch import WarmupLrScheduler
import paddle.nn.functional as F
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

cfg = set_cfg_from_file("for_paddle/lib/bisenetv1_city.py")

def set_optimizer(model):
    
    wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
    optim = paddle.optimizer.Momentum(
        learning_rate=cfg.lr_start,
        parameters=[
        {'params': wd_params,},

        # {'params': wd_params, "learning_rate":cfg.lr_start,"momentum":float(0),"weight_decay":float(0)},
        # {'params': nowd_params,"learning_rate":cfg.lr_start,"momentum":float(0), 'weight_decay': float(0)},
        # {'params': lr_mul_wd_params, 'learning_rate': float(cfg.lr_start * 10),"momentum":float(0),'weight_decay': float(0)},
        # {'params': lr_mul_nowd_params, 'weight_decay': float(0), 'learning_rate': float(cfg.lr_start * 10),"momentum":float(0)},
        ],  
        # parameters=model.parameters(),
        weight_decay=cfg.weight_decay,
        momentum=0.9
         
    )
    return optim


# optim = paddle.optimizer.Momentum(
#             learning_rate=cfg.lr_start,
#             momentum=0.9,
#             parameters=model.parameters(),
#             weight_decay=cfg.weight_decay)
# optim = paddle.optimizer.SGD(
#         cfg.lr_start,
#         parameters=model.parameters(),
#         # epsilon=1e-08,
#         weight_decay=cfg.weight_decay,
#         # momentum=0.9

#     )









if __name__ == "__main__":
    paddle.set_device("cpu")
    loss_paddle_data = ReprodLogger()
    class_num = 19
    model = BiSeNetV1(class_num)
    static_weights = paddle.load('weights/model_final_v1_city_new.pdparams')
    model.set_dict(static_weights)
    model.eval()
    loss = OhemCrossEntropyLoss(0.7)
    optim = set_optimizer(model)
    lr_schdr = WarmupLrScheduler(cfg.lr_start,optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    for i in range(5):
        fake_data = np.load("fake_data/fake_input_data.npy")
        #     print(fake_data.shape)
        fake_data = paddle.to_tensor(fake_data)
        fake_label = np.load("fake_data/fake_input_label.npy")
        #     print(fake_data.shape)
        fake_label = paddle.to_tensor(fake_label)
        

        # forward
        print('start bp...')
        output1,output2,output3 = model(fake_data)
    
        out_loss1 = loss(output1,fake_label)
        out_loss2 = loss(output2,fake_label)
        out_loss3 = loss(output3,fake_label)
        loss_tol = out_loss1 + out_loss2 + out_loss3
        print(out_loss1)
        loss_tol.backward()
        # for name, tensor in model.named_parameters():
        #     # print("------------the grad")
        #     if name[-4:]!="mean" and name[-4:]!="ance":
        #         # print(name)
        #         grad = tensor.grad
        #         # print(name[-4:])
        #         print(name, tensor.grad.shape)
        #     # break
        optim.step()
        optim.clear_grad()
        print(loss_tol)
        # torch.cuda.synchronize()
        # lr_schdr.step()

        print(loss_tol.dtype)
        print(np.array(loss_tol))
    
