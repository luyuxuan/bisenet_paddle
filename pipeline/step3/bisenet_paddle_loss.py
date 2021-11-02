import sys
import paddle   
sys.path.append('.')
import numpy as np
from reprod_log import ReprodLogger
from for_paddle.models.bisenetv1 import BiSeNetV1
from for_paddle.lib.ohem_loss import OhemCrossEntropyLoss
import paddle.nn.functional as F

if __name__ == "__main__":
    paddle.set_device("cpu")
    loss_paddle_data = ReprodLogger()
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
    loss = OhemCrossEntropyLoss(0.7)

    # forward
    print('start inference')
    out = model(fake_data)[0]
    
    out_loss = loss(out,fake_label)
    print(out_loss.dtype)
    loss_paddle_data.add(f"loss", out_loss.numpy())
    print("save the data")
    loss_paddle_data.save("step3/loss_paddle.npy")
    

