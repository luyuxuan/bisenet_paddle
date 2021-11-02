
import cv2
import numpy as np
import paddle.vision.transforms as t
# from lib.sampler import RepeatedDistSampler
import paddle
import os.path as osp
import sys
sys.path.append('.')
import numpy as np
import random
import cv2
from paddle.io import Dataset, DataLoader
import for_torch.lib.transform_cv2 as T
# from traitlets import Any

# from . import preprocess
labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": -1}
]




class BaseDataset(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train',mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115)):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func
        self.mean = mean
        self.std = std
        self.lb_map = None

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(' ')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)
        self.n_cats = 19
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']
        

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        # print(impth, lbpth)
        img, label = self.get_image(impth, lbpth)
        if not self.lb_map is None:
            label = self.lb_map[label]
            # print("findthecity---------label")
        im_lb = dict(im=img, lb=label)
        # print(self.trans_func)
        if not self.trans_func is None:
            # print("train")
            im_lb = self.trans_func(im_lb)
        # tran = t.ToTensor()
        # im = tran(im_lb["im"])
        # im = im_lb["im"]/255
        im = im_lb["im"]
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = paddle.to_tensor(im)
        # print(im.dtype)
        np_y = np.array([255]).astype('float32')
        y = paddle.to_tensor(np_y)
        im = paddle.divide(im,y)
        # y = paddle.to_tensor(np_y)
        # im = torch.from_numpy(im).div_(255)
        dtype = im.dtype
        # print(dtype)
        # print(self.mean)
        mean = paddle.to_tensor(self.mean, dtype=dtype)
        mean = paddle.reshape(mean,[3,1,1])
        # print("llll",mean.shape)
        std = paddle.to_tensor(self.std, dtype=dtype)
        std = paddle.reshape(std,[3,1,1])
        im = paddle.subtract(im,mean)
        im = paddle.divide(im,std)
        # im = im.sub_(mean).div_(std).clone()
        if not im_lb["lb"] is None:
            # print("have the lb")
            lb = paddle.to_tensor(im_lb["lb"]).astype(np.int64).unsqueeze(0)
        return im, lb






        
        # im = im.transpose(2, 0, 1).astype(np.float32)
        # im
        # im = im.dev_(255)

        # # im = im[:, :, ::-1]  
    
        #     # im = im / 255.0
        # im -= self.mean
        # im /= self.std
        # im = im.transpose(2, 0, 1).astype(np.float32)
        # # print("DataLoader",im.shape)
        # # print("DataLoader",label.shape)label.unsqueeze(0)
        # return im, im_lb["lb"].unsqueeze(0)

    def get_image(self, impth, lbpth):
        # image = cv2.imread(impth, cv2.IMREAD_COLOR)
        # label = cv2.imread(lbpth, cv2.IMREAD_GRAYSCALE)
        img, label = cv2.imread(impth)[:, :, ::-1], cv2.imread(lbpth, 0)
        return img, label

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
