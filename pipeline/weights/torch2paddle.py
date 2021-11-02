import numpy as np
import torch
import paddle
map_location=torch.device('cpu')

def get_keys():
    torch_weights = './model_final_v1_city_new.pth'
    torch_dict = torch.load(torch_weights)['state_dict']
    keys = torch_dict.keys()
    for key in keys:
        print(key)

# def Bisenet_transfer():
#     '''
#     转换权重.paddle.nn.BatchNorm2D 包含4个参数weight, bias, _mean, _variance，
#     torch.nn.BatchNorm2d包含4个参数weight, bias, running_mean, running_var, num_batches_tracked，
#     num_batches_tracked 在PaddlePaddle中没有用到，剩下4个的对应关系为
#                         weight -> weight
#                         bias -> bias
#                         _variance -> running_var
#                         _mean -> running_mean
#     '''
#     torch_weights = './model_final_v1_city_new.pth'
#     paddle_weights = "./model_final_v1_city_new.pdparams"
#     torch_dict = torch.load(torch_weights)['state_dict']
#     paddle_dict = {}
#     bn_names = ['running_mean','running_var']
#     for key in torch_dict:
#         weight = torch_dict[key].cpu().detach().numpy()
#         if 'num_batches_tracked' in key:
#             continue
#         elif bn_names[0] in key:
#             key = key.replace('running_mean','_mean')
#         elif bn_names[1] in key:
#             key = key.replace('running_var','_variance')
#         paddle_dict[key] = weight
    
#     paddle.save(paddle_dict, paddle_weights)



# get_keys()
# Bisenet_transfer()