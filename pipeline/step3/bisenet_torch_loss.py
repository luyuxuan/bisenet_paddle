import sys
import paddle   
sys.path.append('.')
import numpy as np 
import torch
from for_torch.models.bisenetv1 import BiSeNetV1
from for_torch.lib.ohem_ce_loss import OhemCELoss
from reprod_log import ReprodLogger


if __name__ == "__main__":
    loss_torch_data = ReprodLogger()
    net = BiSeNetV1(19)
    net.load_state_dict(torch.load('weights/model_final_v1_city_new.pth', map_location='cpu'))
    net.eval()

    fake_data = np.load("fake_data/fake_input_data.npy")
    fake_data = torch.from_numpy(fake_data)
    fake_label = np.load("fake_data/fake_input_label.npy")
    fake_label = torch.from_numpy(fake_label)
    print(fake_data.shape)
    out = net(fake_data)[0]
    loss = OhemCELoss(0.7)


    loss_pre = loss(out,fake_label)
    # print(loss_pre)

    # loss_pre = np.array(loss_pre.detach().numpy).astype(np.float32)
    # print(loss_pre.dtype)
    loss_torch_data.add(f"loss", loss_pre.detach().numpy())
    print("save the data")
    loss_torch_data.save("step3/loss_torch.npy")



