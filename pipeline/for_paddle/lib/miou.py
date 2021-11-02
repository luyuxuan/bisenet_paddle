
import math
import sys
sys.path.append('.')  
import os
import os.path as osp
import cv2
import numpy as np
# from get_dataloader import get_data_loader,build_paddle_data_pipeline
import importlib
import paddle
# from models.bisenetv1 import BiSeNetV1
import tqdm
from reprod_log import ReprodLogger
import time
from PIL import Image



def accuracy(loader,
             model,
             classes=19,
             png_save_dir="images",
             iou_save_dir="logs/mIOU.txt"):
    model.eval()
    data_list = []
    total_len = len(loader)
    for i, (x, label, size, name) in enumerate(loader):
        print(f"{i+1}/{total_len}", end='\r')
        # 防止反向传播的计算
        # input_var = Variable(input, volatile=True)
        input_var = paddle.to_tensor(x, stop_gradient=True)

        # print(input_var.shape)
        output = model(input_var)
        # save seg image
        output = output.cpu().numpy()[0]  # 1xCxHxW ---> CxHxW
        gt = np.asarray(label.numpy()[0], dtype=np.uint8)
        output = output.transpose(1, 2, 0)  # CxHxW --> HxWxC
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

        # output_color = cityscapes_colorize_mask(output)
        output = Image.fromarray(output)
        output.save('%s/%s.png' % (png_save_dir, name[0]))
        # output_color.save('%s/%s_color.png' % (png_save_dir, name[0]))

    iou_mean, iou_list = _get_iou(data_list, classes, save_path=iou_save_dir)
    print("mIou result saved at " + iou_save_dir)
    return iou_mean, iou_list


def get_iou(data_list, classes=19, save_path=None):
    return _get_iou(data_list, classes, save_path)


def _get_iou(data_list, classes=19, save_path=None):
    from multiprocessing import Pool

    ConfM = ConfusionMatrix(classes)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    if save_path:
        with open(save_path, 'a') as f:
            f.write('\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')
    return aveJ, j_list


class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None, ignore_label=255):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))
        self.ignore_label = ignore_label

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == self.ignore_label:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m



def get_round_size(size, divisor=32):
    return [math.ceil(el / divisor) * divisor for el in size]


def calculate_area(pred, label, num_classes, ignore_index=255):

    if len(pred.shape) == 4:
        pred = paddle.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = paddle.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(
                             pred.shape, label.shape))
    pred_area = []
    label_area = []
    intersect_area = []
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = paddle.logical_and(pred == i, mask)
        label_i = label == i
        intersect_i = paddle.logical_and(pred_i, label_i)
        pred_area.append(paddle.sum(paddle.cast(pred_i, "float32")))
        label_area.append(paddle.sum(paddle.cast(label_i, "float32")))
        intersect_area.append(paddle.sum(paddle.cast(intersect_i, "float32")))

    pred_area = paddle.concat(pred_area)
    label_area = paddle.concat(label_area)
    intersect_area = paddle.concat(intersect_area)

    return intersect_area, pred_area, label_area


def mean_iou(intersect_area, pred_area, label_area):
    
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), format(miou,'.8f')



reprod_logger = ReprodLogger()
if __name__ == "__main__":
    # device = paddle.get_device()
    # paddle.set_device(device)
    net = BiSeNetV1(19)
    load_layer_state_dict = paddle.load('/paddle_torch/weight/model_final_v1_city_new.pdparams')
    net.set_dict(load_layer_state_dict)
    # org_aux = net.aux_mode
    # net.aux_mode = 'eval'

    # is_dist = dist.is_initialized()
    _,dl = build_paddle_data_pipeline(cfg, mode='val')
    net.eval()
    data_list = []
    # print(dl.shape)
    diter = enumerate(dl)
        
    for i, (img, labels) in diter:
        print("one_batch_miou")
        output = net(img)[0]
        # print(output.shape)
        gt = np.asarray(labels.detach().numpy()[0], dtype=np.uint8)
        output = output.detach().numpy()[0]
        # output = np.array(output)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        data_list.append([gt.flatten(), output.flatten()])

        mean_iou, per_class_iou = get_iou(data_list)
        print("theresult")
        print(mean_iou)
        data_list.append(mean_iou)
    np.save("./miou.npy",data_list)
    reprod_logger.add("miou", data_list)
    reprod_logger.add("miou", np.array([data_list]))
    reprod_logger.save("./metric_paddle.npy")











def evaluate(cfg, weight_pth):
    # logger = logging.getLogger()

    ## model
    # logger.info('setup and restore model')
    net = BiSeNetV1(19)
    load_layer_state_dict = paddle.load('./model_final_v1_city_new.pdparams')
    
    net.set_dict(load_layer_state_dict)
    heads, mious = eval_model(cfg, net.module)
    # logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
