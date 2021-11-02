import sys
import paddle   
sys.path.append('.')
import numpy as np 
import torch
from for_torch.models.bisenetv1 import BiSeNetV1
from reprod_log import ReprodLogger


if __name__ == "__main__":
    reprod_logger = ReprodLogger()
    net = BiSeNetV1(19)
    net.load_state_dict(torch.load('weights/model_final_v1_city_new.pth', map_location='cpu'))
    net.eval()

    fake_data = np.load("fake_data/fake_input_data.npy")
    fake_data = torch.from_numpy(fake_data)
    print(fake_data.shape)
    out = net(fake_data)[0]
    # print(out)
    # print(out.shape)
    print('save logs')
    reprod_logger.add("logits", out.detach().numpy())
    reprod_logger.save("step1/forward_torch.npy")


