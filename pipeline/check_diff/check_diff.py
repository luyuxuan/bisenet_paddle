import sys
sys.path.append('.')
from reprod_log import ReprodDiffHelper
import numpy as np

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()

#dataset_diff
    # torch_info = diff_helper.load_info("step1/test_dataset_torch.npy")
    # paddle_info = diff_helper.load_info("step1/test_dataset_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="./dataset_diff.log") 


#forward_diff
    # torch_info = diff_helper.load_info("step1/forward_torch.npy")
    # paddle_info = diff_helper.load_info("step1/forward_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="./forward_diff.log") 


#metric_diff
    # torch_info = diff_helper.load_info("step2/metric_torch.npy")
    # paddle_info = diff_helper.load_info("step2/metric_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="metric_diff.log") 


#loss_diff
    # torch_info = diff_helper.load_info("step3/loss_torch.npy")
    # paddle_info = diff_helper.load_info("step3/loss_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="loss_diff.log") 


#lr_diff
    torch_info = diff_helper.load_info("step4/lr_torch.npy")
    paddle_info = diff_helper.load_info("step4/lr_paddle.npy")
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="lr_diff.log") 

#bp_align_diff
    # torch_info = diff_helper.load_info("step4/bp_align_torch.npy")
    # paddle_info = diff_helper.load_info("step4/bp_align_paddle.npy")
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(path="bp_align_diff.log") 
    
    
