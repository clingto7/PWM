---
status: accepted
date: 2026-11-24
---

# 代理画面用网格光栅化和给定高斯两种画法，做对照

首轮的代理画面同时实现两种画法并相互对照：网格光栅化，以及"给每个实体绑定一组不优化的高斯、用 gsplat 渲染"。不使用 DA3 导出的高斯。代理画面不含颜色，只输出深度和实例掩码，光流后置。哪一种成为主线，由对照结果决定。

背景：单张图没有多视角约束，标准的 3DGS 逐场景优化做不了，proposal 也没有要求优化；它议题 2 的选项 B 就是"每个物体由几十上百个高斯组成，直接调用 gsplat 渲染"。先例方面，PWorldModel 的编译器只输出身份、语义、运动方向三张图，不承载外观（论文 4.3 节）。

## Considered Options

- **网格光栅化**（实现并对照）。把高度图和车辆网格当普通网格画。深度和掩码精确，边缘干净。只能画网格，点和粒子画不了。
- **给定高斯加 gsplat**（实现并对照）。地面用高度图的格子，车辆用网格表面采样的圆盘，各自绑定一组不优化的高斯。能画点和粒子，为后续 GNN 的粒子节点和可微渲染留路。已知的风险：圆盘之间可能露缝，边缘处 ID 会混，深度是加权平均。
- **DA3 导出的高斯**（不做）。代理不需要外观，这份数据在这条路线里没有用处。另外按标准 3DGS 格式解析，它的中心落在自身深度图 ±10% 以内的只有 3%；这个现象没有得到解释（未读导出代码），但已不影响决定。
- **"gsplat 为主线、光栅化做对照"**（agent 在第二轮的建议，未作为主从关系采纳）。用户决定两种都试，主从由结果定。

## Consequences

- 对照指标：深度误差、实例掩码 IoU、光流误差、每帧耗时。容差等测量后定。
- 待确认：gsplat 能否直接输出深度和任意数量的附加通道。本次讨论没能核对到原始文档，采用这条画法时第一个确认。
- 这个对照同时回答 proposal 没有论证的问题：为什么用高斯渲染而不用光栅化。

## Accepted addendum (2026-11-26): first-round main renderer

Rasterization is the first-round main renderer and exact reference for depth and
instance IDs. Fixed geometric Gaussians with gsplat remain an optional measured
comparison. Raster depth and ID PNGs retain their actual 16-bit values; color
palettes are used only for preview videos. The comparison can inform a later
renderer decision without changing the first-round acceptance path. The
negative car-size plausibility result remains visible in the run record and
conclusion; choosing rasterization does not repair that reconstruction limit.
