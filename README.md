# Pruning

## 模型剪枝 ™️

#### 1. Torch-Pruning

![](./images/intro.png)

* paper: [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900)
* github: [https://github.com/VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)

关于模型剪枝这边推荐一种**通用的结构化剪枝工具**[Torch-Pruning](https://github.com/VainF/Torch-Pruning),不同于torch.nn.utils.prune中利用掩码(Masking)实现的“模拟剪枝”, Torch-Pruning采用了一种名为DepGraph的非深度图算法, 能够“物理”地移除模型中的耦合参数和通道-->🚀详细讲解请移步官方repo。

(1) install

```
pip install torch-pruning
```

(2) yolov8_prune

```
git clone https://github.com/ultralytics/ultralytics.git 
cp yolov8_pruning.py ultralytics/
cd ultralytics 
git checkout 44c7c3514d87a5e05cfb14dba5a3eeb6eb860e70 # for compatibility
python yolov8_pruning.py
```

注意： 模型训练 ultralytics 版本一定和剪枝版本对应，目前测试版本 ultralytics==8.0.90.

#### 2. Prune
剪枝原理探索

[点我](https://u1g62ufvk30.feishu.cn/docx/MVrfdvgZfo6cQ3xeSkEcBLcKnrf)

Yolov8的通道剪枝方案(bn剪枝):
训练baseline --> 稀疏化训练 --> 剪枝 --> 微调 --> 部署

步骤如下：
1. 训练

2. 稀疏化训练


3. 剪枝


4. 微调


5. onnx


6. engine