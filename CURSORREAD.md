好，这里我给你一版**“收敛后的最终 CursorRead”**。
这不是探索文档，而是：

* 👉 **Cursor 可以逐 step 写代码**
* 👉 **每一步都有明确“不过就停”的判据**
* 👉 **防塌缩、防发散、防误优化写死在流程里**

你可以把它直接命名为：

> `cursor_read.md`

---

# 🧭 cursor_read.md

## Panoramic Multi-view Depth Alignment — Stepwise Implementation Guide

> ⚠️ 规则
> **任何 Step 未通过单元测试与 checklist，禁止进入下一步**
> 本流程假设：
>
> * 相机位姿绝对准确（冻结）
> * DAP 深度连续、单调、低噪声

---

## Step 0：问题与不变量确认（必须先做）

### 目标

明确哪些量 **永远不能被优化**，避免隐式退化。

### 冻结变量

* [ ] 所有相机位姿 ( T_i )
* [ ] 球面投影模型
* [ ] DAP 原始深度值（仅作为输入）

### 明确不做的事

* [ ] 不做 per-pixel 深度自由优化
* [ ] 不优化相机位姿
* [ ] 不对远景做几何对齐

---

### Checklist（不满足直接停）

* [ ] 任意 loss 不包含相机位姿
* [ ] 深度的自由度 < 100 / camera

---

## Step 1：单视角深度重参数化（结构正确性）

### 目标

建立**不会破坏单调性与光滑性**的深度变形模型
**不引入任何跨视角约束**

---

### 实现内容

1. 球面坐标 ((\theta,\phi)) ↔ 像素

2. log-depth 空间

3. 全局单调映射：
   [
   z' = S_i(z),\quad z=\log D^{DAP}
   ]
   （1D 单调 spline）

4. 方向相关缩放：
   [
   s_i(\theta,\phi)=\exp(g_i(\theta,\phi))
   ]
   （SH 或 B-spline grid，低频）

5. 合成深度：
   [
   D'=\exp(z')\cdot s_i
   ]

---

### 单元测试（必须全部通过）

* [ ] **单调性**
  [
  D_a^{DAP} < D_b^{DAP} \Rightarrow D'_a \le D'_b
  ]

* [ ] **正值性**
  [
  D' > 0 \quad \forall (\theta,\phi)
  ]

* [ ] **零变形一致性**

```
S(z)=z, g=0  ⇒  D' == D_DAP
```

* [ ] **连续性**
  相邻像素深度梯度无尖峰

---

### Checklist

* [ ] 未出现 per-pixel 可学习参数
* [ ] g 的幅度被限制（如 |g| < 0.3）
* [ ] spline 至少一个参考点被冻结（防尺度漂移）

---

## Step 2：跨视角几何对齐（不塌缩）

### 目标

引入多视角几何一致性
**但不允许出现常数深度 / 球壳解**

---

### 实现内容

#### 1️⃣ 深度 → 世界点

[
P_i = T_i ( D'_i \cdot r(\theta,\phi) )
]

---

#### 2️⃣ 几何一致性 loss（组合式）

**主约束：Point-to-Ray**
[
L_{p2r}
=======

\left|
(P_i-C_j)
---------

((P_i-C_j)\cdot r_j) r_j
\right|
]

**弱约束：log-depth 一致性**
[
L_{depth}
=========

|
\log D'*j - \log D*{i\rightarrow j}
|
]

---

#### 3️⃣ 远景处理（硬规则）

* 若 ( D^{DAP} \ge 100m )：

  * ❌ 不参与 ( L_{p2r} )
  * ❌ 不参与 ( L_{depth} )
  * ✅ 仅参与平滑项

---

### 单元测试

* [ ] **Identity 测试**
  所有视角深度相同 → 几何 loss ≈ 0

* [ ] **尺度退化测试（关键）**
  手动将所有 D' 设为常数 → loss **显著增大**

* [ ] **位姿冻结测试**
  梯度不回传到 ( T_i )

---

### Checklist

* [ ] ( \lambda_{p2r} \gg \lambda_{depth} )
* [ ] 未使用 ICP / 最近邻
* [ ] 未对远景做几何对齐

---

## Step 3：防退化护栏（必须存在）

### 目标

从优化目标上 **数学性地排除退化最优解**

---

### 必须加入的护栏

#### 1️⃣ log-depth 先验锚点

[
L_{prior}=|\log D' - \log D^{DAP}|^2
]

---

#### 2️⃣ 球面平滑正则

[
L_{smooth}=|\nabla_\theta D'|^2+|\nabla_\phi D'|^2
]

---

#### 3️⃣ 方向变形约束

[
L_g = |g|^2
]

---

### 单元测试（缺一不可）

* [ ] **塌缩复现测试**
  关闭 ( L_{prior} ) → 观察塌缩
  打开 ( L_{prior} ) → 塌缩消失

* [ ] **远景稳定性**
  远景深度不随迭代剧烈变化

* [ ] **能量下降性**
  所有 loss 随迭代稳定下降

---

### Checklist

* [ ] ( L_{prior} ) 权重非零
* [ ] 远景 mask 生效
* [ ] 无方向 scale 爆炸

---

## Step 4：最终联合优化（只允许在此开启）

### 目标

在 **所有护栏生效** 的前提下进行联合优化

---

### 总 loss 形式（固定）

[
L =
\lambda_{p2r} L_{p2r}
+
\lambda_{depth} L_{depth}
+
\lambda_{prior} L_{prior}
+
\lambda_{smooth} L_{smooth}
+
\lambda_{g} L_{g}
]

推荐起步权重：

```
λ_p2r    = 1.0
λ_depth  = 0.1
λ_prior  = 1.0
λ_smooth = 0.01
λ_g      = 0.01
```

---

### 最终检查（不过就回滚）

* [ ] 深度未整体塌缩
* [ ] 未整体推到 100m 球壳
* [ ] 单视角仍光滑连续
* [ ] 多视角结构一致性明显提升

---

## 一句话版 Cursor 原则（写在文件最上面也行）

> **We do not optimize pixels.
> We optimize structured depth deformation.
> Every added degree of freedom must come with an explicit guardrail.**

---

