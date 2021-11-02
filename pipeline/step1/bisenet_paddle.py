import sys
import paddle   
sys.path.append('.')
import numpy as np
from reprod_log import ReprodLogger
from for_paddle.models.bisenetv1 import BiSeNetV1

if __name__ == "__main__":
    paddle.set_device("cpu")
    reprod_logger = ReprodLogger()
    class_num = 19
    model = BiSeNetV1(class_num)
    static_weights = paddle.load('weights/model_final_v1_city_new.pdparams')
    model.set_dict(static_weights)
    model.eval()
    fake_data = np.load("fake_data/fake_input_data.npy")
    # print(fake_data.shape)
    fake_data = paddle.to_tensor(fake_data)
    fake_data = fake_data.astype('float32')
    # forward
    print('start inference')
    out = model(fake_data)[0]
    
    # print(out.shape)
    print('save logs')
    reprod_logger.add("logits", out.numpy())
    reprod_logger.save("step1/forward_paddle.npy")