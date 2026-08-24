# ParkourMoE RA-L 论文草稿说明

这个目录是基于当前代码实现整理的一版 RA-L 风格 LaTeX 草稿，核心文件如下：

- `main.tex`：论文主文件
- `references.bib`：参考文献
- `figures/`：后续补图目录

## 1. Overleaf 使用方式

推荐直接把整个 `paper/parkour_moe_ral` 目录上传到 Overleaf。

当前 `main.tex` 默认使用：

```tex
\documentclass[letterpaper,10pt,conference]{ieeeconf}
```

这样做是为了更接近 RA-L 初稿常用的 IEEE/RAS 版式。

如果 Overleaf 报错缺少 `ieeeconf.cls`，有两种处理方式：

1. 直接在 Overleaf 新建一个 IEEE conference/`ieeeconf` 模板项目，再把这里的 `main.tex` 和 `references.bib` 替换进去。
2. 从官方 RA-L/IEEE 模板包里补充 `ieeeconf.cls` 及相关样式文件后再编译。

说明：

- 我当前本地环境没有安装 LaTeX，所以这次没有办法在本机实际编译验证。
- 论文内容已经尽量按 RA-L 写法收紧，但最终投稿前仍建议你在 Overleaf 上做一次页数、参考文献和浮动体位置检查。
- 按 RA-L 官方说明，Letter 正文是 `6` 页，两栏 IEEE 格式，最多允许 `2` 页付费超页；页数里包含图、表、参考文献和附录。
- RA-L 投稿系统要求选择 `2-5` 个关键词，其中前 `2` 个要从 RA-L 预定义关键词表里选。

## 2. 这版草稿已经写进去的内容

- 当前 `parkour_moe` 的任务定义、观测维度、训练设置
- `ParkourEstimator` 的多模态 token 化与 terrain selector
- `ReadWriteExpert + shared_state` 的核心方法描述
- `mcp_code` 的组成和 actor-critic 的连接方式
- 当前生效的 estimator 损失项与训练策略
- RA-L 风格的实验章节结构

## 3. 你现在必须补的图

建议至少补 4 张主图。

### Figure 1: 方法总览图

位置：`main.tex` 中的 `Figure~\ref{fig:framework}`

建议内容：

- 左侧放输入：`obs_now`、`10x45 proprio history`、`2x58x87 proxy depth`
- 中间放两阶段 estimator：
  - token encoder
  - selector transformer
  - terrain token
  - main transformer
  - router
  - 4 个 read/write experts
  - shared state update
- 右侧放输出：
  - `v_t`
  - `h_tf`
  - `z_mu`
  - `z_tm`
  - `terrain one-hot`
  - actor / critic

这张图最重要，建议你手动画，不要直接截图代码。

### Figure 2: 地形总览图

位置：`main.tex` 中的 `Figure~\ref{fig:terrain_suite}`

建议内容：

- 每类 terrain 选 1 张最有代表性的俯视图或斜视图
- 至少覆盖：
  - `single_gap`
  - `two_row_stones`
  - `one_row_stones`
  - `single_bridge`
  - `air_beams`
  - `air_stones`
  - `hurdle`
  - `ramp`
  - `stairs_up`
  - `flat`
- 最好再补一个 proxy depth 可视化，说明视觉输入长什么样

如果版面不够，可以只放 6-8 类 terrain，再在正文里说明其余类别。

### Figure 3: 定性运动结果图

位置：`main.tex` 中的 `Figure~\ref{fig:qualitative}`

建议内容：

- 选 3 到 4 个典型 terrain
- 每个 terrain 放 3 到 5 帧连续画面
- 推荐优先展示：
  - jump gap
  - crossing beams/stones
  - hurdle
  - stairs/ramp

如果能同时叠加：

- 深度图
- 预测地形图 `m_hat`
- gate 权重条形图

这张图会更有说服力。

### Figure 4: 表征分析图

位置：`main.tex` 中的 `Figure~\ref{fig:tsne}`

建议内容：

- gating weights 的 t-SNE
- latent `z_t` 或 terrain latent 的 t-SNE

仓库里已经有现成脚本：

- `legged_gym/scripts/plot_terrain_gate_zt_tsne.py`

建议至少做两张子图：

1. 按 predicted terrain 着色的 gating t-SNE
2. 按 actual terrain 着色的 latent t-SNE

如果不同 terrain 分簇明显，这张图很适合支持“MoE 确实学到了 terrain-dependent specialization”。

## 4. 你必须补的主实验

论文主表对应 `main.tex` 里的 `Table~\ref{tab:main_results}`。

最推荐的对比是：

1. `Blind PPO baseline`
   - 不使用深度输入
   - 只保留 proprio 控制
   - 用来证明视觉/terrain-aware estimator 的价值

2. `Single expert`
   - 把 `expert_num=4` 改成 `1`
   - 用来证明多专家结构本身有收益

3. `No terrain token`
   - 去掉 terrain selector 和 terrain token conditioning
   - 用来证明显式 terrain-aware routing 的价值

4. `No routing regularization`
   - 关掉 `load_balance_loss` 和 `terrain_swav_loss`
   - 用来证明 gate 不加约束时会退化

5. `No vision-toggle training`
   - easy terrain 上不再周期性翻转 `mask_vision`
   - 用来证明 fallback 训练策略有意义

6. `Full ParkourMoE`
   - 当前完整模型

## 5. 主实验应该报告什么指标

最少要有这 4 个：

- `success rate`
- `normalized forward progress`
- `fall rate`
- `no-progress rate`

如果版面够，再补：

- `command tracking error`
- `mean episode length`
- `termination reason breakdown`

### 更建议的统计方式

- 每个方法至少跑 `3` 个随机种子
- 每类 terrain 单独统计
- 再给一个 overall average

这样主表会比只放 reward 更像 RA-L。

## 6. 必做消融

论文第二张表对应 `main.tex` 里的 `Table~\ref{tab:ablation}`。

优先级最高的消融如下：

1. `w/o terrain selector`
2. `w/o read-write experts`
3. `w/o terrain-map reconstruction`
4. `w/o terrain-ID supervision`
5. `w/o SwAV consistency`
6. `w/o load balancing`
7. `w/o vision-toggle training`

如果你不想做太多，最少保留前 4 个。

## 7. 强烈建议补的鲁棒性实验

这个框架的一个卖点是“视觉可用时用视觉，视觉不可靠时尽量别崩”。

所以建议单独做一个 robustness 小节，比较：

1. 正常 depth
2. depth 全遮挡 / 全零
3. depth 随机 dropout
4. 可选：高噪声 depth

这部分最好只在 easy terrain 上做，因为你当前代码就是在 easy terrain 上训练 vision toggle。

## 8. 关于 selector 的写法建议

当前代码里有一点论文里一定要诚实写清楚：

- 训练时的 `vision toggle` 是真的生效的
- 但推理时根据 `depth autoencoder` 重建误差自动关视觉的那行代码目前是注释掉的

所以论文里不要直接写成：

`we deploy an online selector that disables vision when reconstruction error is high`

除非你后面把这部分重新打开并做了实验。

更稳妥的写法是：

- 当前实现包含一个 AE-based selector interface
- 本文主要验证训练时的 vision masking/fallback 机制
- 在线 selector 作为可选扩展或附加实验

## 9. 投稿前你还要改的地方

至少再检查下面这些内容：

1. 把摘要里的蓝色 TODO 数值补齐
2. 把所有表格中的 `--` 换成真实结果
3. 把所有 placeholder figure 换成正式图片
4. 把作者、单位、致谢改成你的真实信息
5. 检查 `keywords` 是否需要按 RA-L/PaperCept 的标准词表调整
6. 检查是否超页
7. 最后统一语言风格，避免 “current code” 这种工程化表达保留太多

## 10. 这版草稿的定位

这不是“最终投稿版”，而是：

- 已经把方法和实验结构搭好
- 已经尽量贴合当前真实代码
- 只差图、结果和少量润色就能继续往投稿稿件推进的版本

如果你愿意，我下一步可以继续帮你做两件事里的任意一个：

1. 把这篇稿子再压缩成更像正式 RA-L 语气的英文终稿
2. 直接继续补实验表格模板、画图脚本说明，甚至把图 1 的 TikZ 结构图也一起写出来
