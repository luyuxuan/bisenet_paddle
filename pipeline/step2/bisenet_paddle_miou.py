import sys
import paddle   
sys.path.append('.')
import numpy as np
from reprod_log import ReprodLogger
from for_paddle.models.bisenetv1 import BiSeNetV1
from for_paddle.lib.miou import calculate_area,mean_iou
import paddle.nn.functional as F

if __name__ == "__main__":
    paddle.set_device("cpu")
    miou_paddle_data = ReprodLogger()
    class_num = 19
    model = BiSeNetV1(class_num)
    static_weights = paddle.load('weights/model_final_v1_city_new.pdparams')
    model.set_dict(static_weights)
    model.eval()
    fake_data = np.load("fake_data/fake_input_data.npy")
    #     print(fake_data.shape)
    fake_data = paddle.to_tensor(fake_data)
    fake_label = np.load("fake_data/fake_input_label.npy")
    #     print(fake_data.shape)
    fake_label = paddle.to_tensor(fake_label)

    # forward
    print('start inference')
    out = model(fake_data)[0]
    

    N,H,W=fake_label.shape
    probs = paddle.zeros((N, class_num, H, W), dtype=paddle.float32)
    probs += F.softmax(out,axis=1)
    output = paddle.argmax(probs, axis=1)
    print(output.shape)
    intersect_area, pred_area, label_area = calculate_area(output, fake_label, class_num)
    class_iou, miou = mean_iou(intersect_area, pred_area, label_area)
    miou_result = np.array(miou).astype(np.float32)
    miou_paddle_data.add(f"miou", miou_result)
    print("save the data")
    miou_paddle_data.save("step2/metric_paddle.npy")
    

