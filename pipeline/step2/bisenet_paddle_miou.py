import sys
import paddle   
sys.path.append('.')
import numpy as np
from reprod_log import ReprodLogger
from for_paddle.models.bisenetv1 import BiSeNetV1
from for_paddle.lib.miou import calculate_area,mean_iou
import paddle.nn.functional as F
import importlib
from for_paddle.lib.get_dataloader import build_paddle_data_pipeline
import numpy as np
class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


def set_cfg_from_file(cfg_path):
    spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec_loader = spec.loader.exec_module(cfg_file)
    cfg = cfg_file.cfg
    return cfg_dict(cfg)


cfg_paddle = set_cfg_from_file("for_paddle/lib/bisenetv1_city.py")

paddle_dataset, paddle_dataloader = build_paddle_data_pipeline(cfg_paddle, mode='val')
if __name__ == "__main__":
    paddle.set_device("cpu")
    miou_paddle_data = ReprodLogger()
    class_num = 19
    model = BiSeNetV1(class_num)
    static_weights = paddle.load('weights/model_final_v1_city_new.pdparams')
    model.set_dict(static_weights)
    model.eval()
    print('start inference')
    for i,(img,label) in enumerate(paddle_dataloader):
        print("batch:",i)
        out = model(img)[0]
        N,_,H,W=label.shape
        label = paddle.reshape(label,[N,H,W])
        
        probs = paddle.zeros((N, class_num, H, W), dtype=paddle.float32)
        probs += F.softmax(out,axis=1)
        output = paddle.argmax(probs, axis=1)
        intersect_area, pred_area, label_area = calculate_area(output, label, class_num)
        class_iou, miou = mean_iou(intersect_area, pred_area, label_area)
        miou_result = np.array(miou).astype(np.float32)
        print("miou_result:",miou_result)
        miou_paddle_data.add(f"miou_{i}", miou_result)
    print("save the data")
    miou_paddle_data.save("step2/metric_paddle.npy")
    



