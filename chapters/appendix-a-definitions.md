<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的公式与引用。
  术语条目采用「中文名（English Name）」的形式，便于中英对照。
-->
---
prev-chapter: "塑造模型性格与产品"
prev-url: "17-product"
page-title: "附录 A：定义"
search-title: "附录 A：定义"
meta-description: "RLHF、强化学习、语言模型与后训练术语的定义与背景。"
next-chapter: "超越「只是风格」"
next-url: "appendix-b-style"
lectures:
  - video: "https://www.youtube.com/watch?v=MMDNaeIFVy8&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "第 0 讲：前置知识"
---

# 定义

本附录收录 RLHF 流程中频繁使用的全部定义、符号与操作，并对语言模型做一个简要概览——语言模型是本书的主导应用。

## 语言建模概览

多数现代语言模型的训练目标，是以自回归方式学习 token 序列（词、子词或字符）的联合概率分布。
**自回归**的意思很简单：每个下一次预测都依赖序列中之前的实体。
给定 token 序列 $x = (x_1, x_2, \ldots, x_T)$，模型把整条序列的概率分解为条件分布的连乘：

$$P_{\theta}(x) = \prod_{t=1}^{T} P_{\theta}(x_{t} \mid x_{1}, \ldots, x_{t-1}).$$ {#eq:llming}

为了拟合一个能准确预测该分布的模型，目标常常是最大化“由当前模型预测的训练数据似然”。
为此，我们可以最小化负对数似然（NLL）损失：

$$\mathcal{L}_{\text{LM}}(\theta)=-\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T}\log P_{\theta}\left(x_t \mid x_{<t}\right)\right]. $$ {#eq:nll}

实践中，人们用针对每次“下一词预测”的交叉熵损失，做法是把序列中真实的 token 与模型预测的结果作比较。

语言模型有多种架构，在知识、速度及其他性能特性上有不同权衡。
现代 LM——包括 ChatGPT、Claude、Gemini 等——最常使用**仅解码器 Transformer** [@Vaswani2017AttentionIA]。
Transformer 的核心创新是大量使用**自注意力** [@Bahdanau2014NeuralMT] 机制，让模型能直接关注上下文中的概念、并学习复杂映射。
本书通篇（尤其在讲奖励模型的第 5 章）会讨论如何增加新的头、或修改 Transformer 的语言建模（LM）头。
LM 头是最后一层线性投影，把模型的内部嵌入空间映射到 tokenizer 空间（即词表）。
我们会在本书中看到，语言模型不同的“头”可以被用来把模型微调到不同用途——在 RLHF 中，最常见的就是训练奖励模型，第 5 章会重点讲。

## 机器学习

- **Kullback-Leibler（KL）散度（$\mathcal{D}_{\text{KL}}(P || Q)$）**，也叫 KL 散度，是衡量两个概率分布差异的度量。
对定义在同一概率空间 $\mathcal{X}$ 上的离散概率分布 $P$ 与 $Q$，从 $Q$ 到 $P$ 的 KL 距离定义为：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:def_kl}


## 自然语言处理

- **被选中补全（Chosen Completion，$y_c$）**：在多个备选中被选择或被偏好的那个补全，常记作 $y_{chosen}$。

- **补全（Completion，$y$）**：语言模型针对提示生成的输出文本。补全常记作 $y\mid x$。奖励与其他值常按 $r(y\mid x)$ 或 $P(y\mid x)$ 计算。

- **策略（Policy，$\pi$）**：可能补全上的一个概率分布，由 $\theta$ 参数化：$\pi_\theta(y\mid x)$。

- **偏好关系（Preference Relation，$\succ$）**：表示一个补全优于另一个的符号，例如 $y_{chosen} \succ y_{rejected}$。例如，奖励模型预测某个偏好关系的概率 $P(y_c \succ y_r \mid x)$。

- **提示（Prompt，$x$）**：交给语言模型、用以生成回复或补全的输入文本。

- **被拒绝补全（Rejected Completion，$y_r$）**：成对设定中不受青睐的那个补全。

## 强化学习

- **动作（Action，$a$）**：智能体在环境中做出的决策或移动，常表示为 $a \in A$，其中 $A$ 是可能动作的集合。

- **优势函数（Advantage Function，$A$）**：优势函数 $A(s,a)$ 刻画在状态 $s$ 下采取动作 $a$ 相对于平均动作的相对好处。定义为 $A(s,a) = Q(s,a) - V(s)$。优势函数（与价值函数）可以依赖某个具体策略，记作 $A^\pi(s,a)$。

- **折扣因子（Discount Factor，$\gamma$）**：一个标量 $0 \le \gamma < 1$，在回报中按指数方式降低未来奖励的权重，权衡即时收益与长期收益，并保证无穷时域求和的收敛性。有时不使用折扣，等价于 $\gamma=1$。

- **期望奖励优化（Expectation of Reward Optimization）**：RL 的首要目标，即最大化期望累积奖励：

  $$\max_{\theta} \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$ {#eq:expect_reward_opt}

  其中 $\rho_\pi$ 是策略 $\pi$ 下的状态分布，$\gamma$ 是折扣因子。

- **有限时域回报（Finite Horizon Reward，$J(\pi_\theta)$）**：策略 $\pi_\theta$（由 $\theta$ 参数化）的期望有限时域折扣回报，定义为：

  $$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$ {#eq:finite_horizon_return}

  其中 $\tau \sim \pi_\theta$ 表示按策略 $\pi_\theta$ 采样得到的轨迹，$T$ 是有限时域。

- **同策略（On-policy）**：在 RLHF 中，尤其是在 RL 与直接对齐算法的争论里，**同策略**数据的讨论很常见。在 RL 文献中，“同策略”意味着数据*精确地*由智能体的当前形态生成；但在广义的偏好微调文献中，“同策略”被拓展为“来自该模型版本的生成”——例如开始任何偏好微调之前的某个指令微调检查点。在这个语境下，“异策略”可以指后训练中使用的任何其他语言模型生成的数据。

- **策略（Policy，$\pi$）**，在 RLHF 中也叫**策略模型**：在 RL 中，策略是智能体为决定给定状态下应采取什么动作而遵循的策略或规则：$\pi(a\mid s)$。

- **策略条件值（Policy-conditioned Values，$[]^{\pi(\cdot)}$）**：在 RL 的推导与实现中，一个关键组成部分是“在特定策略条件下收集数据或值”。本书通篇会在价值函数的简化记号（$V,A,Q,G$）与其具体的策略条件值（$V^\pi,A^\pi,Q^\pi$）之间切换。期望值计算中同样关键的是从数据 $d$ 采样，而该数据以特定策略为条件，记作 $d_\pi$（例如在估计 $\mathbb{E}_{s\sim d_\pi,\,a\sim\pi(\cdot\mid s)}\!\left[A^\pi(s,a)\right]$ 时，$s \sim d_\pi$ 且 $a \sim \pi(\cdot\mid s)$）。

- **Q 函数（Q-Function，$Q$）**：估计“在给定状态下采取某个具体动作”的期望累积奖励的函数：$Q(s,a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]$。

- **奖励（Reward，$r$）**：一个标量值，指示某个动作或状态的可取程度，通常记作 $r$。

- **状态（State，$s$）**：环境当前的构型或处境，通常记作 $s \in S$，其中 $S$ 是状态空间。

- **轨迹（Trajectory，$\tau$）**：轨迹 $\tau$ 是智能体经历的状态、动作与奖励的序列：$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)$。

- **轨迹分布（Trajectory Distribution，$(\tau\mid\pi)$）**：策略 $\pi$ 下一条轨迹的概率为 $P(\tau\mid\pi) = p(s_0)\prod_{t=0}^T \pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)$，其中 $p(s_0)$ 是先验状态分布，$p(s_{t+1}\mid s_t,a_t)$ 是转移概率。

- **价值函数（Value Function，$V$）**：估计“从给定状态出发”的期望累积奖励的函数：$V(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$。

## 仅属 RLHF

- **参考模型（Reference Model，$\pi_{\text{ref}}$）**：RLHF 中使用的一组已保存参数，其输出被用来正则化优化过程。

## 扩展术语表

- **思维链（Chain-of-Thought，CoT）**：思维链是语言模型的一种特定行为：被引导成把问题按一步一步的形式拆解。它最初的形态是通过提示 “Let's think step-by-step” [@wei2022chain] 实现的。

- **蒸馏（Distillation）**：蒸馏是训练 AI 模型的一整套通用实践：让一个模型在更强模型的输出上训练。它是一类已知能造出强大的小模型的合成数据。多数模型会在许可协议（对开放权重模型）或服务条款（对只能通过 API 访问的模型）中明确蒸馏的规则。“蒸馏”这个词如今还承载了 ML 文献中一个具体的技术定义。

- **上下文学习（In-context Learning，ICL）**：这里的“上下文”指语言模型上下文窗口内的任何信息，通常是加到提示里的信息。上下文学习最简单的形式，是在提示之前加入若干同形式的示例。更高级的版本能学会为特定用例选择该纳入哪些信息。

- **（教师—学生）知识蒸馏（(Teacher-student) Knowledge Distillation）**：从特定教师到学生模型的知识蒸馏，是上述蒸馏的一种具体类型，也是这个术语的起源。它是一种具体的深度学习方法：修改神经网络的损失，使其从教师模型在多个潜在 token/logits 上的对数概率中学习，而不是直接从选定的输出学习 [@hinton2015distilling]。用知识蒸馏训练的现代模型系列例子包括 Gemma 2 [@team2024gemma] 与 Gemma 3。对语言建模设定，下一词损失函数可以修改如下 [@agarwal2024policy]，其中学生模型 $P_\theta$ 从教师分布 $P_\phi$ 学习：

$$\mathcal{L}_{\text{KD}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T} P_{\phi}(x_t \mid x_{<t}) \log P_{\theta}(x_t \mid x_{<t})\right]. $$ {#eq:knowledge_distillation}

- **合成数据（Synthetic Data）**：指任何由另一个 AI 系统产出的、用于训练 AI 模型的数据。它可以是模型针对开放式提示生成的文本，也可以是模型对既有内容的改写。
