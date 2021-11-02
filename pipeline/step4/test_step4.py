import sys
sys.path.append('.')
from for_paddle.lib.lr_sch import WarmupLrScheduler
from for_torch.lib.lr_scheduler import WarmupPolyLrScheduler
import torch
import paddle
from reprod_log import ReprodLogger
import numpy as np

if __name__ == "__main__":
    lr_troch_logger = ReprodLogger()
    lr_paddle_logger = ReprodLogger()
    model_torch = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim_torch = torch.optim.SGD(model_torch.parameters(), lr=1e-5)
    model_paddle = paddle.nn.Conv2D(3, 16, 3, 1, 1)
    optim_paddle = paddle.optimizer.SGD(learning_rate=1e-5,parameters=model_paddle.parameters())



    lr = 1e-5
    max_iter = 30
    lr_scheduler_torch = WarmupPolyLrScheduler(optim_torch, 0.9, max_iter, 200, 0.1, 'linear', -1)
    lr_scheduler_paddle = WarmupLrScheduler(lr,optim_paddle, 0.9, max_iter, 200, 0.1, 'linear', -1)

    lrs_torch = []
    lrs_paddle = []
    for _ in range(max_iter):
        lr_torch = lr_scheduler_torch.get_lr()[0]
        lrs_torch.append(lr_torch)
        lr_paddle = lr_scheduler_paddle.get_lr()[0]
        lrs_paddle.append(lr_paddle)
    lr_troch_logger.add("lr",np.array(lr_torch))
    lr_paddle_logger.add("lr",np.array(lr_paddle))
    print("save the data...")
    lr_troch_logger.save("step4/lr_torch.npy")
    lr_paddle_logger.save("step4/lr_paddle.npy")
