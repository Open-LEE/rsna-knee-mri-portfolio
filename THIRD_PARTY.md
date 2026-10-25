# 来源、归属与分发范围

本仓库是竞赛复现与实验工程项目，不是全部模型从零开发的声明。

| 组件 | 来源 | 项目中的角色 |
|---|---|---|
| 比赛任务与数据 | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection | 数据必须按比赛条款自行取得；不随仓库分发 |
| 报告弱标签 | https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels | 训练监督来源，非本项目独立生成的全量标签 |
| DINOv2 | https://github.com/facebookresearch/dinov2 | 通用预训练视觉编码器 |
| timm DINOv2 Small | https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m | 通过 timm 加载的模型实现与权重 |
| R6/R7/CoAtNet 融合参考 | https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters | 公开强基线与融合方案 |
| R7 checkpoint 参考 | https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956 | checkpoint 复现参考 |
| CoAtNet 权重 | https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007 | 公共模型，不是本项目训练成果 |

完整权重挂载链接见 docs/REPRODUCE.md。第三方内容遵循各自授权，不能仅因可下载就当成可任意重新授权。本地 R6 数据元信息的许可字段为 `Other (specified in description)`，因此本包不包含该权重及完整嵌入式第三方提交代码。报告标签的 CC0 许可曾由参赛者反馈，未把这一声明扩展到竞赛影像、报告或其他模型。

仓库包含本项目会话开发的 starter、DINO 训练/评估、缓存与融合参考实现。保留原依赖的导入关系；PyTorch、timm、pydicom 等依赖需另行安装。实现使用 AI 编程辅助。

没有给整个项目套用一个涵盖所有上游资产的统一许可证。当前源码用于项目展示；若计划对原创部分另行授予开源许可证，应由仓库所有者确认，并保持第三方资产许可独立。
