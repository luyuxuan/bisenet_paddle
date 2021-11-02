
from get_dataloader import get_data_loader,build_torch_data_pipeline
import importlib
import paddle
import numpy as np
from reprod_log import ReprodLogger
from tqdm import tqdm
import torch
from models.bisenetv1 import BiSeNetV1
from lr_scheduler import WarmupPolyLrScheduler
n_classes = 19

import torch.nn.functional as F


def miou(imgs,label,net,n_classes):
    N,H,W = label.shape
    news = []
    hist = torch.zeros(n_classes, n_classes)
    probs = torch.zeros((N, n_classes, H, W), dtype=torch.float32)
    logits = net(imgs)[0]
    logits = F.interpolate(logits, size=(H,W),
                        mode='bilinear', align_corners=True)
    probs += torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    keep = label != 255
    print(keep.shape)
    print(preds.shape)
    hist += torch.bincount(label[keep] * n_classes + preds[keep],minlength=n_classes ** 2).view(n_classes, n_classes)
    ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
    # print(ious)
    ious = np.array(ious)
    for i in range(len(ious)):
        print(ious[i].dtype)
        if ious[i]>0:
            news.append(ious[i])
        else:
            news.append(float(0))
    # ious = np.array(ious)
    print("iou_class",news)
    miou = np.mean(news)
    print(miou)
    return format(miou,'.8f')









class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


def set_cfg_from_file(cfg_path):
    spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec_loader = spec.loader.exec_module(cfg_file)
    cfg = cfg_file.cfg
    return cfg_dict(cfg)
import math
def get_round_size(size, divisor=32):
    return [math.ceil(el / divisor) * divisor for el in size]
from ohem_ce_loss import OhemCELoss
loss = OhemCELoss(0.7)








cfg = set_cfg_from_file("/Users/luyuxuan/Downloads/BiSeNet-master/configs/bisenetv1_city.py")

# a = get_data_loader(cfg, mode='train')
# b = get_data_loader(cfg, mode='val')
logger_torch_data = ReprodLogger()
# c = build_paddle_data_pipeline(cfg, mode='train')
torch_dataset, torch_dataloader = build_torch_data_pipeline(cfg, mode='val')
nnn = np.array(len(torch_dataset))
print(nnn)
import torch

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




net = BiSeNetV1(19,aux_mode='train')
net.load_state_dict(torch.load('/Users/luyuxuan/Desktop/paddle_torch/weight/model_final_v1_city_new.pth', map_location='cpu'))
net.train()
# optim = set_optimizer(net)

# ## lr scheduler
# lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
#     max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
#     warmup_ratio=0.1, warmup='exp', last_epoch=-1,)


# optim = torch.optim.SGD(net.parameters(),
#                                     lr=0.001,
#                                     momentum=0.9,
#                                     weight_decay=5e-4)
optim = torch.optim.SGD(
        # params_list,
        net.parameters(),
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        # weight_decay=5e-4
    )
loss_torch_data = ReprodLogger()
# for i in range(3):
#     print("star bp ing.........")
#     diter = enumerate(tqdm(torch_dataloader))

#     fake_data = np.load("/Users/luyuxuan/Desktop/paddle_torch/step_1/fake_loss_input_data.npy")
#     #     print(fake_data.shape)

#     fake_label = np.load("/Users/luyuxuan/Desktop/paddle_torch/step_1/fake_loss_input_label.npy")
#     a = torch.from_numpy(fake_data)
#     # label = np.random.randint(1,12,size=[1,3,3])
#     b = torch.from_numpy(fake_label)
#     optim.zero_grad()
#     # print(lr)
#     logits,logits2,logits3 = net(a)
#     # print(logits)
#     loss_pre = loss(logits, b)
#     loss_pre2 = loss(logits2, b)
#     loss_pre3 = loss(logits3, b)
#     loss_tol = loss_pre + loss_pre2 + loss_pre3
#     loss_tol.backward()
#     # for name, tensor in net.named_parameters():
#     #     grad = tensor.grad
#     #     print(name, tensor.grad.shape)
#         # break
#     optim.step()
#     # torch.cuda.synchronize()
#     # lr_schdr.step()
#     # print(loss_tol)
#     loss_pre = loss_pre.detach().numpy() 
#     loss_tol = loss_tol.detach().numpy() 
#     print(loss_pre)
#     print(loss_tol)
    
#     loss_torch_data.add(f"loss_{i}", loss_tol.detach().numpy())
# loss_torch_data.save("./bp_align_pytorch.npy")
# miou_result = miou(a,b,net,19)

# miou_result = np.array(miou_result).astype(np.float32)
# print(miou_result.dtype)
# miou_torch_data = ReprodLogger()
# miou_torch_data.add(f"miou", np.array(miou_result))
# miou_torch_data.save("./miou_torch.npy")


# logits, *logits_aux = net(a)
# # print(lb.shape)



# hist = torch.zeros(n_classes, n_classes)
# # diter = enumerate(tqdm(dl))

# # for i, (imgs, label) in diter:
# N,H, W = b.shape
# label = b.squeeze(1)
# size = label.size()[-2:]
# probs = torch.zeros(
#         (N, n_classes, H, W), dtype=torch.float32)

# sH, sW = int(H), int( W)
# sH, sW = get_round_size((sH, sW))
# im_sc = F.interpolate(a, size=(sH, sW),
#         mode='bilinear', align_corners=True)

# im_sc = im_sc
# logits = net(im_sc)[0]
# print(logits.shape)
# logits = F.interpolate(logits, size=size,
#         mode='bilinear', align_corners=True)
# probs += torch.softmax(logits, dim=1)
    
# preds = torch.argmax(probs, dim=1)
# keep = label != 255
# hist += torch.bincount(
#     label[keep] * n_classes + preds[keep],
#     minlength=n_classes ** 2
#     ).view(n_classes, n_classes)
# print("one,batch,miou",np.nanmean((hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())).detach().numpy()))
# # if dist.is_initialized():
# #     dist.all_reduce(hist, dist.ReduceOp.SUM)
# ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
# miou = np.nanmean(ious.detach().numpy())
# print(miou)





















# loss_pre = loss(logits, b)
# print(loss_pre)
# loss_pre = np.array(loss_pre).astype(np.float32)
# print(loss_pre.dtype)
# loss_torch_data = ReprodLogger()
# loss_torch_data.add(f"loss", loss_pre)
# loss_torch_data.save("./loss_torch.npy")











for idx in range(5):
    # rnd_idx = np.random.randint(0, len(paddle_dataset))

    logger_torch_data.add(f"dataset_{idx}",
                            torch_dataset[idx][0].detach().numpy())

for idx, (torch_batch) in enumerate(torch_dataloader):
    if idx >= 5:
        break
    
    logger_torch_data.add(f"dataloader_{idx}",torch_batch[0].detach().numpy())
print("save the npy")
logger_torch_data.save("./pipeline/step_1/test_dataset_torch.npy")



