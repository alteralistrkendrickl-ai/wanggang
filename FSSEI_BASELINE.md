# FS-SEI外部基线适配说明

## 定位

本基线来源于公开仓库：

- https://github.com/BeechburgPieStar/FS-SEI
- 关联论文：*Few-Shot Specific Emitter Identification via Deep Metric Ensemble Learning*

公开仓库的PyTorch目录提供了STC-CVCNN、Triplet Loss、Center Loss和少样本LR测试代码，但缺少其入口引用的`get_adsb_dataset.py`与项目级`utils.py`。其编码器末端还固定为`Linear(576, feature_dim)`，对应约4800点输入，不能直接处理WiSig的256点IQ记录。因此，本项目只能将其登记为“FS-SEI repository adapted STC-CVCNN”，不能声称完整复现DMEL论文。

## 明确改动

1. 保留九个复数一维卷积、批归一化、ReLU和逐级池化结构。
2. 将固定长度Flatten头替换为全局平均池化，使编码器接受256点输入。
3. 使用公开PyTorch实现对应的目标：交叉熵 + 0.01×Batch-Hard Triplet + 0.01×Center Loss。
4. 修复公开PyTorch入口未将Center参数交给优化器的问题。
5. 使用现有WiSig功率归一化，预训练仅访问90类base train/val。
6. 下游仅访问`validation_novel/train`与`validation_novel/val`；最终test保持封存。

## 公平比较边界

- 它是外部少样本SEI架构基线，不是跨接收机或跨日期方法。
- 它与P3MC/A1使用相同的WiSig strict身份划分、相同shot集合和相同支持集随机种子。
- 正式结果必须单独报告“adapted implementation”，不得直接引用原论文在ADS-B上的准确率作数值对比。
- 只有通过单元测试、双协议3轮训练和N=10验证后，才进入正式多训练种子比较。

## 入口

训练：

```bash
python wisig_train_fssei.py --protocol cross-rx --epochs 3 --seed 2024 --device cuda
```

验证：

```bash
python wisig_validate_lr.py \
  --protocol cross-rx \
  --encoder fssei-stc \
  --checkpoint-path /path/to/best_encoder.pth \
  --shots 1 5 10 15 20 \
  --iterations 10 \
  --base-seed 2024 \
  --device cuda
```
