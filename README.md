# BiseNetV1


# 一、简介

[BiSeNetV1](https://arxiv.org/abs/1808.00897): Bilateral Segmentation Network for Real-time Semantic Segmentation是由旷世的一篇实时语义分割网络，发表在ECCV2018，网络以2048*1024的输入，在Cityscapes 数据集上取得了68.4% Mean IOU ，同时速度达到105 FPS。




# 二、论文精度


mIOUs and fps on cityscapes val set:
| none | ss | ssc | msf | mscf | fps(fp16/fp32) | link |
|------|:--:|:---:|:---:|:----:|:---:|:----:|
| bisenetv1 | 75.44 | 76.94 | 77.45 | 78.86 | 68/23 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v1_city_new.pth) |



mIOUs on cocostuff val2017 set:
| none | ss | ssc | msf | mscf | link |
|------|:--:|:---:|:---:|:----:|:----:|
| bisenetv1 | 31.49 | 31.42 | 32.46 | 32.55 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v1_coco_new.pth) |

Tips: 
**ss** means single scale evaluation, **ssc** means single scale crop evaluation, **msf** means multi-scale evaluation with flip augment, and **mscf** means multi-scale crop evaluation with flip evaluation. 



# 三、数据集

使用的数据集是 [Cityscapes](https://aistudio.baidu.com/aistudio/datasetdetail/64550) 

### 下载方法一

[Cityscapes](https://www.cityscapes-dataset.com/) 上下载gtCoarse、gtFine、leftImg8bit文件，解压该文件，将该数据集根据官方repo转化为[19](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py) 个类型。处理完成后对应的文件夹应该具有如下的结构：

```
├── infer.list
├── train.list
├── test.list
├── trainval.list
├── val.list
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val

```

后将解压的文件放入 `data/` 文件夹下。

# 四、环境依赖

- 硬件：CPU、GPU
- 框架：PaddlePaddle ≥ 2.0.0(本实验均在2.2版本下)

# 五、快速开始

### step1: clone

```
# clone this repo
git clone https://github.com/632652101/CGNet-PP.git
cd FOR_BISENET
```

### step2: 安装依赖

```
pip install -r requirements.txt
```

### step3: 论文复现打卡点
```
cd pipeline
```

```
├── check_diff
│   └── check_diff.py #查看复现过程中不同的模块
├── dataset
├── fake_daya
│   ├── fake_input_data.npy 
│   ├── fake_input_label.npy
│   └── generate_fake.py
├── for_paddle
│   ├── lib  #包含数据集定义，loss，miou，config
│   ├── models  #网络结构
├── for_torch
│   ├── lib
│   ├── models
├── step1
│   ├── bisenet_paddle.py
│   ├── bisenet_torch.py
│   └── test_step1.py
├── step2
│   ├── bisenet_paddle_miou.py
│   └── bisenet_torch_miou.py
├── step3
│   ├── bisenet_paddle_loss.py
│   └── bisenet_torch_loss.py
├── step4
│   ├── bp_paddle.py
│   ├── bp_torch.py
│   └── test_step1.py
├── step5
│   ├── train.py
│   └── test_step5.py
├── weights
├── dataset_diff.log
├── forward_diff.log
├── loss_diff.log
├── lr_diff.log
├── metric_diff.log
├── bp_diff.log

```

# 六、其他

### 复现日志文件

复现的日志文件存放在logs/log_reprod文件夹下。



