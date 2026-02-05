# MultiPanoramaDepthRefine
好，我给你一版**“收敛逻辑清楚、风险说清、决策明确”的 README 重写版**。
这版是**总结 + 收敛后的最终设计**，不是探索文档，适合：

* 放在 repo 根目录
* 给合作者 / 评审 / 未来的你快速理解
* 作为 Cursor 后续实现的**最高层约束**

---

# 📘 README.md

## Panoramic Multi-view Depth Alignment with Smooth Direction-aware Deformation

---

## 1. 做什么（What）

本项目解决以下问题：

> 在 **已知且精确的全景相机位姿** 下，
> 对来自 DAP（Depth Any Panoramas）的 **多视角预测深度图** 进行联合优化，使其：

* 在 **多视角之间几何一致**
* 在 **单视角内部保持光滑、连续、单调**
* **不引入高频噪声或局部碎裂**
* 对 **远景（≥100m）不强制几何一致，仅保持平滑**

输入：

* 3–4 个全景相机位姿（固定，不优化）
* 对应的等轴柱状 RGB
* 对应的 DAP 深度图

输出：

* 对齐后的多视角深度图
* 可选：一致的稠密点云，用于 3DGS / BA / 渲染

---

## 2. 为什么（Why）

### 2.1 DAP 深度的关键特性（前提假设）

DAP 预测深度具有以下结构性特点：

* 深度场 **连续、光滑、几乎无噪声**
* 深度值 **满足单调性（近小远大）**
* 存在 **系统性几何畸变**（非随机噪声）
* 远景深度被 **统一截断为 100m 球壳**

因此：

* ❌ 不适合 per-pixel 深度自由优化（会破坏光滑性）
* ❌ 不适合重新训练深度网络
* ❌ 不适合将深度直接当作真实 SfM 点云

---

### 2.2 正确的建模视角

本项目采用的核心假设是：

> **每一张 DAP 深度图不是“错误的深度”，
> 而是真实深度在球面方向空间上，
> 经过低频、单调、方向相关变形后的结果。**

因此，优化目标应当是：

> **恢复这些“深度变形函数”，
> 而不是直接修改深度值本身。**

---

## 3. 怎么做（How）

### 3.1 问题定义（数学形式）

对第 ( i ) 个全景相机：

* 位姿：
  [
  \mathbf{T}_i \in SE(3)
  ]

* 球面方向参数：
  [
  (\theta,\phi)\in[-\pi,\pi)\times\left[-\frac{\pi}{2},\frac{\pi}{2}\right]
  ]

* DAP 深度：
  [
  D_i^{\mathrm{DAP}}(\theta,\phi)\in(0,D_{\max}],\quad D_{\max}=100\text{m}
  ]

---

### 3.2 深度重参数化（核心建模）

不直接优化 ( D )，而是定义结构化变形模型：

[
D_i'(\theta,\phi)
=================

\exp!\Big(
S_i(\log D_i^{\mathrm{DAP}}(\theta,\phi))
\Big)
\cdot
s_i(\theta,\phi)
]

其中：

#### （1）全局单调映射 ( S_i )

* 定义在 log-depth 空间
* 1D 单调 spline
* 修正整体尺度与非线性压缩
* 保证深度单调性不被破坏

#### （2）方向相关缩放 ( s_i(\theta,\phi) )

* 定义在球面上
* 低频参数化（球谐或 B-spline grid）
* 捕捉全景方向相关的系统性畸变
* 采用正参数化 ( s_i=\exp(g_i) )，并施加幅度约束

---

### 3.3 跨视角几何一致性（联合约束）

#### 点云生成

[
\mathbf{P}_i(\theta,\phi)
=========================

\mathbf{T}_i
\big(
D_i'(\theta,\phi),\mathbf{r}(\theta,\phi)
\big)
]

---

#### 组合几何约束（关键总结）

本项目**不使用单一几何误差**，而是采用互补组合：

1. **Point-to-Ray 距离（主约束）**
   [
   L_{\mathrm{p2r}}
   ]

* 对尺度误差鲁棒
* 约束方向一致性
* 适合全景小基线场景

2. **Ray-space 深度一致性（弱约束）**
   [
   L_{\mathrm{depth}}
   =
   \big|
   \log D_j' - \log D_{i\rightarrow j}
   \big|
   ]

* 提供径向信息
* 权重显著低于 p2r

---

### 3.4 防止退化解的“硬护栏”（核心总结）

多视角对齐在理论上存在以下退化最优解：

* 所有深度趋于常数
* 所有点塌缩到相机中心
* 所有点被推到 100m 球壳

本项目通过 **显式设计** 防止上述情况：

#### 1️⃣ log-depth 先验锚点

[
L_{\mathrm{prior}}
==================

|\log D' - \log D^{\mathrm{DAP}}|^2
]

#### 2️⃣ 远景去几何化

* ( D^{\mathrm{DAP}}\ge100\text{m} ) 的像素：

  * 不参与跨视角几何对齐
  * 仅参与平滑正则

#### 3️⃣ 有界的方向变形

* ( s_i(\theta,\phi)=\exp(g_i) )
* 对 ( g_i ) 加 L2 正则与幅度限制

#### 4️⃣ 全局尺度冻结

* 固定 spline 的参考点
  或
* 强制近景 log-depth 统计量保持不变

---

### 3.5 总体优化目标

[
\min
;
\lambda_{\mathrm{p2r}} L_{\mathrm{p2r}}
+
\lambda_{\mathrm{depth}} L_{\mathrm{depth}}
+
\lambda_{\mathrm{prior}} L_{\mathrm{prior}}
+
\lambda_{\mathrm{smooth}} L_{\mathrm{smooth}}
+
\lambda_{\mathrm{far}} L_{\mathrm{far}}
]

其中：

* ( \lambda_{\mathrm{p2r}} \gg \lambda_{\mathrm{depth}} )
* ( \lambda_{\mathrm{prior}} ) 提供全局锚点
* 远景不参与几何对齐

---

## 4. 设计原则总结（一句话版）

* **不优化像素，只优化结构**
* **不相信远景几何**
* **方向一致性优先于径向一致性**
* **鲁棒性与尺度锚点必须同时存在**
* **任何新增自由度都必须配套护栏**

---

如果你愿意，下一步我可以：

* 把这版 README **直接拆解成 Cursor 的 Step1/2/3 实现提示**
* 或给你一版 **“最小可跑实现 + 单元测试骨架”**
* 或专门写一节 **Failure modes & Debug Guide**

你告诉我你想继续哪一步就行。
