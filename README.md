# P3MC：少样本特定辐射源识别

本项目实现论文 **P3MC: Dual-Level Data Augmentation for Robust Few-Shot Specific Emitter Identification**，并加入可选的 AWGN 信道增强和轻量特征解耦模块（LFDB）。

## 方法概览

预训练阶段同时优化：

1. 相位旋转分类（`rot_cls`）
2. 发射设备分类（`sei_cls`）
3. 特征空间 Manifold Mixup（`mml`）
4. 可选的指纹/非指纹特征解耦（`decouple`）

下游阶段冻结预训练编码器，在每类只有少量样本的情况下使用 Linear、Logistic Regression、KNN 或 SVM 分类。

## 环境

推荐 Python 3.8+。安装依赖：

```bash
pip install -r requirements.txt
```

## 数据目录

ADS-B 数据文件默认放在：

```text
Datasets/ADS-B/
├── X_train_90Class.npy
├── Y_train_90Class.npy
├── X_test_90Class.npy
├── Y_test_90Class.npy
└── ...
```

项目中已经包含 ADS-B 的 10、20、30、90 类数据文件。每个 IQ 样本应为 `(2, length)`，代码使用前 4800 个采样点。

## 预训练

默认运行 CVTSLANet、8 个旋转类别、随机 0～30 dB AWGN：

```bash
python pretext.py
```

常用参数：

```bash
python pretext.py -e CVTSLANet --epoch 100 --rot_num 8
python pretext.py --use_lfdb --lfdb_weight 1.0
python pretext.py --no_awgn
python pretext.py --awgn_min 5 --awgn_max 25
python pretext.py --resume runs/Pretext_random_rot/.../checkpoint.pth
```

输出位于：

```text
runs/Pretext_random_rot/<实验名>/
├── best_encoder.pth
├── final_encoder.pth
├── checkpoint.pth
└── <时间戳>/
```

## 下游实验

统一入口：

```bash
# 冻结编码器并训练线性分类头
python downstream.py -c linear -s 5

# 传统分类器
python downstream.py -c lr -s 1 5 10
python downstream.py -c knn -s 5
python downstream.py -c svm -s 5

# 指定测试信噪比
python downstream.py -c lr -s 5 --snr_enable --snr 0 10 20
```

结果会保存到 `runs/Downstream_random_rot/`，每次随机少样本采样的准确率写入 Excel。

## 主要文件

```text
pretext.py                    预训练入口
downstream.py                 下游统一入口
downstream_linear.py          PyTorch 线性探针
downstream_lr.py              Logistic Regression
downstream_knn.py             KNN
downstream_svm.py             SVM
utils/config.py               参数和路径配置
utils/get_dataset.py          数据加载、归一化、少样本采样
utils/channel_aug.py          AWGN 信道增强
models/CVTSLANetFeature.py    主要特征编码器
models/lfdb.py                可选特征解耦模块
```

## 可用编码器

- `CVTSLANet`
- `CVTSLANet-Shallow`
- `CVTSLANet-Deep`
- `ResNet18`
- `STC`
- `CVCM`

## 说明

- 下游实验默认从 `runs/Pretext_random_rot/<实验名>/best_encoder.pth` 加载权重。
- 若只测试随机初始化编码器，可设置 `--pretrain_epoch 0`。
- LFDB 默认关闭，保证与原始 P3MC 基线一致；使用 `--use_lfdb` 显式启用。
- 验证集不添加随机 AWGN，避免验证指标随随机噪声波动。

## 引用

```bibtex
@ARTICLE{11073147,
  author={Xu, Lai and Zhang, Weijie and Tang, Tiantian and Zhang, Qianyun and Lin, Yun and Xuan, Qi and Gui, Guan},
  journal={IEEE Internet of Things Journal},
  title={P3MC: Dual-Level Data Augmentation for Robust Few-Shot Specific Emitter Identification},
  year={2025},
  volume={12},
  number={18},
  pages={38668-38679},
  doi={10.1109/JIOT.2025.3586923}
}
```
