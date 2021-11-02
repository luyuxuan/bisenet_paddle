import importlib
import sys
sys.path.append('.')
from for_torch.lib.get_dataloader import build_torch_data_pipeline
from for_paddle.lib.get_dataloader import build_paddle_data_pipeline
import numpy as np
from reprod_log import ReprodLogger,ReprodDiffHelper
logger_paddle_data = ReprodLogger()
logger_torch_data = ReprodLogger()
class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


def set_cfg_from_file(cfg_path):
    spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec_loader = spec.loader.exec_module(cfg_file)
    cfg = cfg_file.cfg
    return cfg_dict(cfg)




diff_helper = ReprodDiffHelper()
cfg_torch = set_cfg_from_file("for_torch/lib/bisenetv1_city.py")
cfg_paddle = set_cfg_from_file("for_paddle/lib/bisenetv1_city.py")



torch_dataset, torch_dataloader = build_torch_data_pipeline(cfg_torch, mode='val')
paddle_dataset, paddle_dataloader = build_paddle_data_pipeline(cfg_paddle, mode='val')






for idx in range(5):
    rnd_idx = np.random.randint(0, len(paddle_dataset))
    logger_paddle_data.add(f"dataset_{idx}",
                            paddle_dataset[rnd_idx][0].numpy())
    logger_torch_data.add(f"dataset_{idx}",
                            torch_dataset[rnd_idx][0].detach().numpy())

for idx, (paddle_batch, torch_batch
            ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
    if idx >= 5:
        break
    logger_paddle_data.add(f"dataloader_{idx}", paddle_batch[0].numpy())
    logger_torch_data.add(f"dataloader_{idx}",
                            torch_batch[0].detach().numpy())

diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
diff_helper.report()








# for idx in range(5):  
#     # rnd_idx = np.random.randint(0, len(paddle_dataset))
#     logger_paddle_data.add(f"dataset_{idx}",paddle_dataset[idx][0].numpy())
# for idx, (paddle_batch) in enumerate(paddle_dataloader):
#     if idx >= 5:
#         break
#     logger_paddle_data.add(f"dataloader_{idx}", paddle_batch[0].numpy())
# print("save the npy")
# logger_paddle_data.save("./pipeline/step_1/test_dataset_paddle.npy")




# for idx in range(5):
#     # rnd_idx = np.random.randint(0, len(paddle_dataset))
#     logger_torch_data.add(f"dataset_{idx}",
#                             torch_dataset[idx][0].detach().numpy())

# for idx, (torch_batch) in enumerate(torch_dataloader):
#     if idx >= 5:
#         break
    
#     logger_torch_data.add(f"dataloader_{idx}",torch_batch[0].detach().numpy())
# print("save the npy")
# logger_torch_data.save("./pipeline/step_1/test_dataset_torch.npy")