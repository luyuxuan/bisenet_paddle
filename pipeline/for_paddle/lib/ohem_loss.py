
import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np


class OhemCrossEntropyLoss(nn.Layer):

    def __init__(self, thresh=0.7, min_kept=10000, ignore_index=255):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):

        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)

        # get the label after ohem
        n, c, h, w = logit.shape
        label = label.reshape((-1, ))
        valid_mask = (label != self.ignore_index).astype('int64')
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = F.softmax(logit, axis=1)
        prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask)

            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose((1, 0))
            prob = prob * label_onehot
            prob = paddle.sum(prob, axis=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index.numpy()[0])
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).astype('int64')
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask

        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index

        label = label.reshape((n, 1, h, w))
        valid_mask = valid_mask.reshape((n, 1, h, w)).astype('float32')
        loss = F.softmax_with_cross_entropy(
            logit, label, ignore_index=self.ignore_index, axis=1)
        loss = loss * valid_mask
        avg_loss = paddle.mean(loss) / (paddle.mean(valid_mask))

        # label.stop_gradient = True
        # valid_mask.stop_gradient = True
        # print(avg_loss.numpy())
        print(avg_loss.dtype)
        return avg_loss
        
if __name__ == "__main__":
    fake_data_data = np.load("./real_real_pred_1.8086490631103516.npy")
    fake_data_label = np.load("./real_real_label_1.8086490631103516.npy")
    a = paddle.to_tensor(fake_data_data)
    # # label = np.random.randint(1,12,size=[1,3,3])
    b = paddle.to_tensor(fake_data_label)
    # a = paddle.randn([1,19,512,1024])
    # b = paddle.randn([4,1,512,1024])
    # b = np.random.random(size=(1,512,1024)).astype(np.int64) 

    b = paddle.to_tensor(b)
    print(a.shape)
    print(b.shape)
    loss = OhemCrossEntropyLoss()

    loass = loss(a,b)
    print("llllll",loass)


