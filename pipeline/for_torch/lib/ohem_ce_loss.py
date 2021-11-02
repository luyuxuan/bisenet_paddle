


import torch
import torch.nn as nn
import torch.nn.functional as F




import torch
import torch.nn as nn
import torch.nn.functional as F


#  import ohem_cpp
#  class OhemCELoss(nn.Module):
#
#      def __init__(self, thresh, ignore_lb=255):
#          super(OhemCELoss, self).__init__()
#          self.score_thresh = thresh
#          self.ignore_lb = ignore_lb
#          self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='mean')
#
#      def forward(self, logits, labels):
#          n_min = labels[labels != self.ignore_lb].numel() // 16
#          labels = ohem_cpp.score_ohem_label(
#                  logits, labels, self.ignore_lb, self.score_thresh, n_min).detach()
#          loss = self.criteria(logits, labels)
#          return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        # print(n_min)
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        # print(torch.mean(loss_hard).dtype)
        return torch.mean(loss_hard)




# class OhemCELoss(nn.Module):

#     def __init__(self, thresh, ignore_lb=255):
#         super(OhemCELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
#         self.ignore_lb = ignore_lb
#         self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

#     def forward(self, logits, labels):
#         n_min = labels[labels != self.ignore_lb].numel() // 16
#         loss = self.criteria(logits, labels).view(-1)
#         loss_hard = loss[loss > self.thresh]
#         if loss_hard.numel() < n_min:
#             loss_hard, _ = loss.topk(n_min)
#         return torch.mean(loss_hard)

import numpy as np
if __name__ == '__main__':
    loss = OhemCELoss(0.7)
    fake_data_data = np.load("/Users/luyuxuan/Desktop/paddle_torch/real_real_pred_1.8086490631103516.npy")
    fake_data_label = np.load("/Users/luyuxuan/Desktop/paddle_torch/real_real_label_1.8086490631103516.npy")
    a = torch.from_numpy(fake_data_data)
    # # label = np.random.randint(1,12,size=[1,3,3])
    b = torch.from_numpy(fake_data_label)
    print(a.shape)
    print(b.shape)
    
    # a = torch.randn([1,2,3,3])
    # label = np.random.randint(1,12,size=[1,3,3])

    # label = torch.tensor([1,3,3])
    # lb = torch.squeeze(label, 1)
    nnnn = loss(a,b)
    print(nnnn)


