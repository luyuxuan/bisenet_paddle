import sys
import paddle   
sys.path.append('.')
import numpy as np 
import torch
from for_torch.models.bisenetv1 import BiSeNetV1
from for_torch.lib.eval import miou
from reprod_log import ReprodLogger


if __name__ == "__main__":
    miou_torch_data = ReprodLogger()
    net = BiSeNetV1(19)
    net.load_state_dict(torch.load('weights/model_final_v1_city_new.pth', map_location='cpu'))
    net.eval()

    fake_data = np.load("fake_data/fake_input_data.npy")
    fake_data = torch.from_numpy(fake_data)
    fake_label = np.load("fake_data/fake_input_label.npy")
    fake_label = torch.from_numpy(fake_label)
    print(fake_data.shape)
    out = net(fake_data)[0]

    miou_result = miou(out,fake_label,19)
    miou_result = np.array(miou_result).astype(np.float32)
    # print(miou_result.dtype)
    miou_torch_data.add(f"miou", np.array(miou_result))
    print("save the data")
    miou_torch_data.save("step2/metric_torch.npy")



