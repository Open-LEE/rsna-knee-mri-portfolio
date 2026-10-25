# 复现指南

## 环境和范围

建议 Python 3.11+，GPU 训练使用 Kaggle T4 ×2。自行安装匹配环境的 PyTorch / torchvision，再安装根目录 requirements.txt。训练固定 timm 1.0.29；Kaggle 容器的其他版本以当次日志为准，仓库没有提供完整环境锁文件，不承诺不同平台逐位相同。

无需模型和比赛数据的检查：

```bash
python scripts/check_project.py
```

这是 CPU 合成数据测试及 Notebook 结构检查，不是模型质量测试。

## Kaggle 训练路径

### 第一步：冻结 DINO

导入 `notebooks/rsna_dino_branch_train_v1.ipynb`，设置 GPU T4 ×2，Internet On（首次下载通用预训练权重）。添加：

1. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
2. https://www.kaggle.com/code/openll/rsna20261010/output
3. https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels

选择包含 4,407 例缓存的版本，Save Version → Save & Run All。输出 `dino-branch-v1/`，包含通用 `dino_encoder.pt`、冻结特征、`split.csv`、头权重和指标。挂载链接是否可访问取决于原作者可见性；如访问失败，需自行生成对应输入，不能用空文件替代。

### 第二步：部分微调对照

分别导入 `rsna_dino_finetune_v2.ipynb` 与 `rsna_dino_four_block_v3.ipynb`，两个都从第一步冻结实验的同一版本开始，不把两层微调接着训练成四层。

Input：

1. https://www.kaggle.com/code/openll/rsna20261010/output
2. https://www.kaggle.com/code/openlm/dinov2-12/output

设置 GPU T4 ×2、Internet On，Save & Run All。若使用自己重跑的 v1，请添加自己的完整 Output。旧实验输出用于复核，复现实验应保持所选版本固定。

### 第三步：固定三折验证

导入 `rsna_dino_group_cv_v1.ipynb`，添加与第二步相同的两个 Input。GPU T4 ×2、Internet On，Save & Run All。用通用编码器和重新初始化的头训练，不能替换成任务微调权重。

输出 `dino-group-cv-v1/` 中的三个 `fold*/final_model.pt`、OOF、划分及聚合指标。小结果包不包含权重。比赛期间对应的训练页面：

https://www.kaggle.com/code/openlm/dinov2-v1/output

该地址由参赛者提供，实际复现应核对输出文件结构与权重，不能仅按 Notebook 名称判断。

## 命令行训练

以下从仓库的 `src` 目录运行；路径为示例，需替换成自己的实际路径。

```bash
cd src
python -m knee.preprocess --data /data/rsna --split train --out /data/mri-cache --workers 2
python dino_branch_train.py --data /data/rsna --weak /data/report_labels_v2.csv --cache /data/mri-cache --out /outputs/dino-branch-v1
python dino_finetune_four_blocks.py --base /outputs/dino-branch-v1 --cache /data/mri-cache --out /outputs/dino-four-block
python dino_group_cv.py --base /outputs/dino-branch-v1 --cache /data/mri-cache --out /outputs/dino-group-cv-v1
```

弱标签文件必须含 `StudyInstanceUID` 和各类别 `__verdict` 列。三折的 base 是冻结 v1，不是四层模型。每个输出使用新目录，以免误用旧实验结果。

## 最终公开集成

本仓库没有重新分发整个第三方推理包。公开参考：

- https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters
- https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956

其所需 Input：

1. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
2. https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406-w1
3. https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss
4. https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007
5. https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004

比赛时以 GPU T4 ×2、Internet Off 执行离线提交。现在比赛已结束，这些步骤用于研究复现，不承诺还能获得正式评分。三折 2% 负结果版本：

https://www.kaggle.com/code/openlm/r6-r7-dino-2?scriptVersionId=361037787

## 发布到 GitHub

在 GitHub 创建空仓库，建议名称 `rsna-knee-mri-portfolio`。解压本项目，在该目录执行：

```bash
git init
git add .
git commit -m "Document RSNA MRI experiments and reproducible training code"
git branch -M main
git remote add origin https://github.com/Open-LEE/rsna-knee-mri-portfolio.git
git push -u origin main
```

向自己的仓库推送时，请替换远程仓库地址。发布前阅读 THIRD_PARTY.md；不要把本地原始数据、报告、权重或凭据目录追加到仓库。项目仓库：https://github.com/Open-LEE/rsna-knee-mri-portfolio 。
