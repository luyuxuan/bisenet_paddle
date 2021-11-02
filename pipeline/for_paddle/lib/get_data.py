
# from torch import float32
from get_dataloader import get_data_loader,build_paddle_data_pipeline
import importlib
import paddle
import numpy as np
from reprod_log import ReprodLogger
from ohem_loss import OhemCrossEntropyLoss
from lr_sch import WarmupLrScheduler
import sys
sys.path.append('../../')  
import paddle.nn.functional as F
from models.bisenetv1 import BiSeNetV1
from eval import get_iou,calculate_area,mean_iou
class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


def set_cfg_from_file(cfg_path):
    spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec_loader = spec.loader.exec_module(cfg_file)
    cfg = cfg_file.cfg
    return cfg_dict(cfg)





cfg = set_cfg_from_file("/Users/luyuxuan/Desktop/paddle_torch/pipeline/step_2/bisenet.py")

a = get_data_loader(cfg, mode='train')
b = get_data_loader(cfg, mode='val')
# c = build_paddle_data_pipeline(cfg, mode='train')
paddle_dataset, paddle_dataloader = build_paddle_data_pipeline(cfg, mode='val')
nnn = np.array(len(paddle_dataset))
print(nnn)

class_num = 19
model = BiSeNetV1(class_num,aux_mode='train')
static_weights = paddle.load('/Users/luyuxuan/Desktop/paddle_torch/weight/model_final_v1_city_new.pdparams')
model.set_dict(static_weights)
model.train()
loss = OhemCrossEntropyLoss(0.7)
# diter = enumerate(paddle_dataloader)

# def set_optimizer(model):
    
#     wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
#     print(len(wd_params), len(nowd_params), len(lr_mul_wd_params), len(lr_mul_nowd_params))
#     #     params_list = [
#     #         {'params': wd_params, },
#     #         {'params': nowd_params, 'weight_decay': 0},
#     #         {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
#     #         {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
#     #     ]
#     # else:
#     #     wd_params, non_wd_params = [], []
#     #     for name, param in model.named_parameters():
#     #         if param.dim() == 1:
#     #             non_wd_params.append(param)
#     #         elif param.dim() == 2 or param.dim() == 4:
#     #             wd_params.append(param)
#     #     params_list = [
#     #         {'params': wd_params, },
#     #         {'params': non_wd_params, 'weight_decay': 0},
#     #     ]
    
#     optim = paddle.optimizer.Momentum(
#         learning_rate=cfg.lr_start,
#         parameters=[
#             {'params': wd_params },
#             {'params': nowd_params, 'weight_decay': float(0)},
#             {'params': lr_mul_wd_params, 'learning_rate': 0.1,},
#             {'params': lr_mul_nowd_params, 'weight_decay': float(0), 'learning_rate':0.1}
#         ],
#         weight_decay=5e-4,
#         momentum=0.9


#     )
#     return optim


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        # print(nowd_params[:8])
        print(len(wd_params), len(nowd_params), len(lr_mul_wd_params), len(lr_mul_nowd_params))
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        print("gggg")
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


optim = paddle.optimizer.Momentum(
            learning_rate=cfg.lr_start,
            momentum=0.9,
            parameters=model.parameters(),
            weight_decay=cfg.weight_decay)

# optim = set_optimizer(model)
# optim = paddle.optimizer.SGD(
#         cfg.lr_start,
#         parameters=model.parameters(),
#         # epsilon=1e-08,
#         weight_decay=cfg.weight_decay,
#         # momentum=0.9

#     )
# lr_schdr = WarmupLrScheduler(cfg.lr_start,optim, power=0.9,
#         max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
#         warmup_ratio=0.1, warmup='exp', last_epoch=-1,)


# loss_paddle_data = ReprodLogger()
# for i in range(3):
#     fake_data = np.load("/Users/luyuxuan/Desktop/paddle_torch/step_1/fake_loss_input_data.npy")
#     #     print(fake_data.shape)
#     fake_data = paddle.to_tensor(fake_data)

#     fake_label = np.load("/Users/luyuxuan/Desktop/paddle_torch/step_1/fake_loss_input_label.npy")
#     #     print(fake_data.shape)
#     fake_label = paddle.to_tensor(fake_label)
    
#     output1,output2,output3 = model(fake_data)
#     # print(output1)
    
#     out_loss1 = loss(output1,fake_label)
#     out_loss2 = loss(output2,fake_label)
#     out_loss3 = loss(output3,fake_label)
#     loss_tol = out_loss1 + out_loss2 + out_loss3
#     print(out_loss1)
#     loss_tol.backward()
#     # for name, tensor in model.named_parameters():
#     #     # print("------------the grad")
#     #     if name[-4:]!="mean" and name[-4:]!="ance":
#     #         # print(name)
#     #         grad = tensor.grad
#     #         # print(name[-4:])
#     #         print(name, tensor.grad.shape)
#     #     # break
#     optim.step()
#     optim.clear_grad()
#     print(loss_tol)
#     # torch.cuda.synchronize()
#     # lr_schdr.step()

#     print(loss_tol.dtype)
    # print(np.array(loss_tol))
   
#     loss_paddle_data.add(f"loss_{i}", loss_tol.numpy())
# loss_paddle_data.save("./bp_align_paddle.npy")







# data_list = [] 
# /Users/luyuxuan/Desktop/paddle_torch/step_1/fake_loss_input_data.npy
# fakedata = np.load

# hist = paddle.zeros(19, 19)
# N,H,W=fake_label.shape
# probs = paddle.zeros((N, 19, H, W), dtype=paddle.float32)
# probs += F.softmax(output,axis=1)
# output = paddle.argmax(probs, axis=1)
# print(output.shape)
# # output = paddle.to_tensor(output)
# intersect_area, pred_area, label_area = calculate_area(output, fake_label, 19)
# class_iou, miou = mean_iou(intersect_area, pred_area, label_area)
# miou_result = np.array(miou).astype(np.float32)
# print(miou_result.dtype)
# # print(miou_result.dtype)
# miou_paddle_data = ReprodLogger()
# miou_paddle_data.add(f"miou", miou_result)
# miou_paddle_data.save("./miou_paddle.npy")
# print("theresult")
# print(miou)








# out_loss = loss(output,fake_label)
# print(out_loss.dtype)
# loss_paddle_data = ReprodLogger()
# loss_paddle_data.add(f"loss", out_loss)
# loss_paddle_data.save("./loss_paddle.npy")

# for i, (img, labels) in diter:
# # for i,(img,labels) in dl:
#     # images = paddle.to_tensor(img)
#     # labels = paddle.to_tensor(label).astype('int64')
#     print("one_batch_miou")
#     print(img.shape)
#     print(labels.shape)
#     output = model(img)[0]
#     out_loss = loss(output,labels)
#     print(out_loss)







logger_paddle_data = ReprodLogger()
for idx in range(5):
    
    # rnd_idx = np.random.randint(0, len(paddle_dataset))
    logger_paddle_data.add(f"dataset_{idx}",paddle_dataset[idx][0].numpy())


for idx, (paddle_batch) in enumerate(paddle_dataloader):
    if idx >= 5:
        break
    logger_paddle_data.add(f"dataloader_{idx}", paddle_batch[0].numpy())
    

print("save the npy")
logger_paddle_data.save("./pipeline/step_1/test_dataset_paddle.npy")



