<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
  代码块与引文保留原文。
-->
---
prev-chapter: "过度优化"
prev-url: "14-over-optimization"
page-title: 正则化
search-title: "第 15 章：正则化"
meta-description: "在不损害基座模型的前提下，让 RLHF 与后训练更新保持有用的正则化方法。"
next-chapter: "评估"
next-url: "16-evaluation"
lectures:
  - video: "https://www.youtube.com/watch?v=IwpYxANrpUs&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=15"
    label: "第 10 讲：RL 中的正则化、RL 为何泛化、SFT 为何遗忘"
---

# 正则化

本书中我们学到了许多修改模型、让它从人类偏好、可验证奖励及其他有价值信号中学习的工具。
我们用到的所有方法都很强大，可能让模型相对上一训练阶段那个强大而通用的模型（常称为参考模型）改变得太多。
当模型从给定奖励中学得太多、导致分布外性能下降时，这被称为“过度优化”（已在上一章讨论）。

在整个 RLHF 优化过程中，会用到许多正则化步骤来防止奖励模型的过度优化。
这些语境下的过度优化表现为模型输出毫无意义的文本。
优化“脱轨”的一些例子包括：模型输出可读的数学推理但答案极端错误、重复文本、切换语言，或过度使用特殊字符。
本章覆盖用来控制模型优化的各种方法。

截至 2026 年最流行的变体——也是大多数 RLHF 实现所用的——是在生成样本上，当前策略相对参考策略的 KL 距离。
“KL 距离”是一个口语化的说法，用来表达训练过程中的*优化距离*；尽管 KL 散度——衡量两个概率分布分离程度的底层数学方法——并不满足成为真正距离度量所需的正式性质（只是把那个数字叫做“距离”，比叫做“分布差异的数值度量”更方便）。
文献中涌现过许多其他正则化技术，随后又在那一脉研究的下一代模型迭代中消失。
也就是说，**核心的“生成上的 KL 距离”之外的正则化，常常是用来稳定实验设置的，而这些设置可以在下一代中被简化掉**。
尽管如此，理解 RLHF 中约束优化的工具仍然很重要。

*本章通篇用 $x$ 表示提示、$y$ 表示补全。这一记号在语言模型文献中很常见——那些方法作用于完整的“提示—补全”对，而非单个 token。*

在带奖励模型 $r_\theta$ 的 RLHF 框架中，一般形式如下：

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {#eq:rl_start}

而参考实现是：

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi_{\text{RL}}(y \mid x) \, \| \, \pi_{\text{ref}}(y \mid x) \right)
$$ {#eq:kl_standard}

## RL 优化中的 KL 散度

数学定义见附录 A「定义」。
KL 散度衡量一个概率分布偏离另一个有多远——KL 为零时，两个分布产生完全相同的输出。
回忆它的定义：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:kl_distance_regularization}

在 RLHF 中，关心的两个分布常常是新模型版本的分布（记作 $P(x)$）与参考策略的分布（记作 $Q(x)$）。
不同优化器使用不同的 KL 方向。本书通篇最常用的那个“KL 惩罚”，称为相对参考策略的**反向 KL**。实践中，它化简为一个蒙特卡洛估计：从 RL 模型采样 token，并从参考模型计算概率。直观上，这个反向 KL 有一个数值特性：当新模型 $P$（即 $\pi_{\text{RL}}$）把大量概率质量放在原始参考模型赋予低概率的地方时，施加很大的惩罚。

另一个 KL 方向在 ML 中仍常被使用，例如某些 RL 算法的内部信任域计算。这个惩罚直观上是在“新模型的更新**没有**把概率施加到 $Q$（即 $\pi_{\text{ref}}$）的高似然区域”时对其进行惩罚。它更接近蒸馏或行为克隆所用的目标。

### 参考模型与生成内容

KL 惩罚最常见的实现方式，是比较训练期间生成的 token 与一个静态参考模型之间的距离。
其直觉是：你正在训练的那个模型，其风格是你希望保持接近的。
这个参考模型最常见的是指令微调模型，也可以是先前的某个 RL 检查点。
简单代入后，我们采样的对象就变成 $\pi_{\text{RL}}(x)$ 与 $\pi_{\text{ref}}(x)$，如 @eq:kl_standard 所示（在标准定义中常写作 $P$ 与 $Q$，用于 RL 的 KL 惩罚时也是如此）。
这样的 KL 散度惩罚早在大型语言模型流行之前就被用于对话智能体 [@jaques2017sequence]，但 KL 控制很快被确立为微调预训练模型的核心技术 [@jaques2020human]。

### 实现示例

实践中，KL 散度的实现常常是**近似**的 [@schulman2016klapprox]，这使实现简单得多。
按上面的定义，当我们直接从分布 $P$ 采样时，KL 的求和可以转成期望（这里的 $x$ 是样本空间上的通用随机变量，不是本书其他地方用的“提示”记号）。
这种情况下，$P$ 是当前被训练模型的生成分布（即不是参考模型）。
于是 KL 散度的计算变为：

$$
\mathcal{D}_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$ {#eq:kl_expectation}

这种基于样本的形式实现起来简单得多，尤其当直接处理语言模型训练中频繁使用的对数概率时。

```python
# Step 1: generate() autoregressively samples a full sequence token by token
generated_tokens = model.generate(inputs)

# Step 2: forward() runs a single pass over the sequence to get per-token logits (no sampling)
logits       = model.forward(generated_tokens[:, :-1]).logits
ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits

# Step 3: Convert logits to log-probabilities
logprobs     = F.log_softmax(logits, dim=-1)
ref_logprobs = F.log_softmax(ref_logits, dim=-1)

# Step 4: Gather the probability each model assigns to the tokens that were actually generated
token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# Step 5: Sum to get sequence-level log-probs; their difference approximates KL
seq_logprob     = token_logprobs.sum(dim=-1)
ref_seq_logprob = ref_token_logprobs.sum(dim=-1)

kl_approx = seq_logprob - ref_seq_logprob
kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

一些示例实现包括 [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) 与 [Hamish Ivison 的 JAX 代码](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)。


## 控制优化的其他工具

在后训练文献中，许多著名模型都包含了其他正则化方法，帮助它们在自己的设定下达到领先性能。
这里举这些例子，是为了展现“一些领先模型如何调整后训练设置以获得稳定优化”，而不是把它们当作在每个设定中都必然有效的工具。
还有无数更有创意的方案可以奏效，也会被发现！

### RL 中的预训练梯度

看待正则化的另一种方式是：你可能有一个希望模型保持接近的*数据集*，正如 InstructGPT 所做的 [@ouyang2022training]“为了修复在公开 NLP 数据集上的性能回退”。
为实现这一点，他们修改了 RLHF 的训练目标。
从 @eq:rl_start 出发，我们可以把它转成一个待优化目标函数：从 RL 策略模型采样、从 RLHF 所用 RL 数据集中的提示 $x$ 采样补全 $y$，得到：
$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right]
$$ {#eq:objective_regularization}

然后，对一批从预训练语料（或其他数据集）中采样的文档，为“在预训练所用的标准自回归下一词预测损失上获得更高概率”追加一份额外奖励，以维持文本连贯性：

$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi_{\text{RL},\theta}(x)) \right]
$$ {#eq:objective_pretraining}

### DPO 中的下一词准确率

近期工作提出用一项负对数似然来平衡直接偏好优化（DPO）的优化 [@pang2024iterative]。
考虑到 DPO 损失的成对性质，同样的损失改动也可以施加到奖励模型训练上，约束模型去预测准确的文本。

该优化是对 DPO 的一处修改。
$$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}}(c_i^w, y_i^w, c_i^l, y_i^l \mid x_i) + \alpha \mathcal{L}_{\text{NLL}}(c_i^w, y_i^w \mid x_i)
$$ {#eq:dpo_nll}

$$
= -\log \sigma \left( \beta \log \frac{P_\theta(c_i^w, y_i^w \mid x_i)}{P_{\text{ref.}}(c_i^w, y_i^w \mid x_i)} - \beta \log \frac{P_\theta(c_i^l, y_i^l \mid x_i)}{P_{\text{ref.}}(c_i^l, y_i^l \mid x_i)} \right) - \alpha \frac{\log P_\theta(c_i^w, y_i^w \mid x_i)}{|c_i^w| + |y_i^w|},
$$ {#eq:dpo_nll_expanded}

其中 $P_{\theta}$ 是可训练的策略模型，$P_{\text{ref.}}$ 是固定的参考模型（通常是 SFT 检查点），$(c_i^w, y_i^w)$ 与 $(c_i^l, y_i^l)$ 表示提示 $x_i$ 的胜出与落败补全。
第一项是标准的 DPO logistic 损失：它用对数似然比之差 $\log \tfrac{P_{\theta}}{P_{\text{ref.}}}$ 来拉大胜者与败者之间的间隔，$\beta$ 控制这个偏好信号把模型从参考模型拉开的强度。
第二项是对胜出补全的、按长度归一化的负对数似然惩罚，权重为 $\alpha$；它有助于让被偏好的文本在绝对的语言建模意义上保持高似然，而不只是相对胜过被拒绝样本。

### 奖励建模中基于间隔的正则化

在 RLHF 技术栈的其他部分，控制优化定义得没那么明确。
大多数奖励模型除了标准的对比式损失之外没有正则化。
直接对齐算法则通过 $\beta$ 参数以不同方式处理相对 KL 散度的正则化（见[直接对齐那一章](https://rlhfbook.com/c/08-direct-alignment)）。

Llama 2 为奖励模型训练提出了一个间隔损失 [@touvron2023llama]：

$$
\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)
$$ {#eq:margin_loss}

其中 $m(y_c, y_r)$ 是两个数据点 $y_c$ 与 $y_r$ 之间的间隔，表示两位标注者评分之差的数值。
这可以通过让标注者在数值刻度上给输出打分来实现，也可以用某种量化的排序方法，例如[李克特量表](https://en.wikipedia.org/wiki/Likert_scale)。

奖励间隔在直接对齐文献中被大量使用，例如奖励加权 DPO；奖励感知偏好优化（RPO）把奖励模型分数按 DPO 损失的方式整合进更新规则 [@adler2024nemotron]；以及 REBEL [@gao2024rebel]，它在回归损失形式中带一个奖励差值权重。

## 隐式正则化

本章其他小节描述的是**显式**正则化：实践者有意加到训练目标里的 KL 惩罚、预训练梯度、间隔损失。
越来越多的实证工作揭示：基于 RL 的后训练还提供了**隐式**正则化——一种内建于同策略优化结构本身的、对记忆与灾难性遗忘的抵抗力。
这源自损失更新的性质，即便没有任何用于控制 RL 训练的显式工具（如 KL 惩罚或回放缓冲区）也是如此。

### SFT 记忆，RL 泛化

后训练社区面对的一个核心问题是：在单个任务上训练时，模型学到的是一个可迁移到未见变体的通用规则，还是记住了训练分布的表面模式？
Chu 等 2025 [@chu2025sft] 用一项受控实证研究回答了这个问题，直接分离出后训练方法——SFT 与 RL——对分布外（OOD）泛化的影响。
答案是明确的：**RL 学到可迁移的规则，而 SFT 记住训练数据，并在分布偏移下崩塌。**

该研究使用两个内建规则变体的环境来理解这些权衡：

- **GeneralPoints** 是一个算术卡牌游戏：模型拿到四张扑克牌，必须用运算符（+、-、*、/）组合它们的数值以凑到目标数（默认 24）。OOD 测试改变的是人头牌的计分方式：训练用一种规则（J、Q、K 都算 10），评估用另一种（J = 11、Q = 12、K = 13）。

- **V-IRL** 是一个真实世界的视觉导航任务：模型遵循语言指令在城市街道中穿行，沿途识别地标。OOD 偏移把动作空间从绝对方向（北、东）切换到相对方向（左、右）。

在所有任务变体上，随着训练算力扩大，RL 持续改善 OOD 性能，而 SFT 尽管在同分布上变好，OOD 性能却持续*下降*。
分化的幅度惊人：在 V-IRL 的纯语言输入设定下（OOD 偏移是从绝对方向坐标到相对方向坐标），RL 把 OOD 逐步准确率从 80.8% 提升到 91.8%，而 SFT 把它从 80.8% 崩到 1.3%。
SFT 模型不只是没能泛化：它还**摧毁了基座模型本来具备的空间推理能力**，退化成一张从指令短语到绝对方向的查找表。

### 在做的过程中保持：同策略数据缓解遗忘

上一节表明，在单个任务上 RL 泛化而 SFT 记忆。
Chen 等 2025 [@chen2025retainingdoingroleonpolicy] 追问了互补的问题：在多个任务上*顺序*训练时，模型能保住它已经知道的东西吗？
他们发现，RL 在目标任务上取得相当或更高的收益，同时遗忘显著少于 SFT；并把这一优势追溯到两种目标所优化之对象的根本差异。

要理解两种方法为何表现如此不同，可以从 KL 散度的视角来看它们的目标。
本节我们首先说明这两种常见的后训练方法可以对应到 KL 散度的两个方向，然后解释“把它们用作损失函数时的数值行为”如何转化为不同的模型行为。

KL 散度定义为两个分布之间对数比的期望 $\mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right]$，它可以写成对数差的形式，并且有两个方向：

- **前向 KL**：$\text{KL}(P \| Q) = \mathbb{E}_{x \sim P}\!\left[\log P(x) - \log Q(x)\right]$
- **反向 KL**：$\text{KL}(Q \| P) = \mathbb{E}_{x \sim Q}\!\left[\log Q(x) - \log P(x)\right]$

其中 $P$ 是目标分布，$Q$ 是我们用参数 $\theta$ 建模的分布。
关键差别在于我们从哪个分布采样：前向 KL 从目标（或最优）分布 $P$ 采样，反向 KL 从我们的策略 $Q$ 采样。
在下面的推导中，$P$ 对应目标 $\pi_\star$（分析 SFT 时是训练数据分布，分析 RL 时是奖励最优策略），$Q$ 对应学到的策略 $\pi_\theta$（我们正在训练的东西）。
SFT 把目标放在前面——$\text{KL}(\pi_\star \| \pi_\theta)$——而 RL 把顺序反过来——$\text{KL}(\pi_\theta \| \pi_\star)$——从而改变了采样的来源。
样本提供用于学习的数据；而目标（SFT 或 RL）用这些数据塑造模型。

#### SFT 的前向 KL

从前向 KL 的定义出发：

$$
\text{KL}(\pi_\star \| \pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) - \log \pi_\theta(y \mid x) \right]
$$

把对数差的期望拆成两项：

$$
= \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) \right] - \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\theta(y \mid x) \right]
$$

第一项 $\mathbb{E}\!\left[\log \pi_\star(y \mid x)\right]$ 只依赖数据分布，等于负熵 $-H(\pi_\star)$——一个不随 $\theta$ 变化的常数。
第二项 $-\mathbb{E}\!\left[\log \pi_\theta(y \mid x)\right]$ 是数据集上的负对数似然，也就是标准的 SFT 交叉熵损失 $\mathcal{L}_\text{SFT}(\theta)$。代入：

$$
= \underbrace{-H(\pi_\star)}_\text{const} + \mathcal{L}_\text{SFT}(\theta) \propto \mathcal{L}_\text{SFT}(\theta)
$$ {#eq:sft_forward_kl}

由于熵项相对 $\theta$ 是常数，两个损失共享相同的梯度与相同的最小值点——**最小化 SFT 损失等价于最小化前向 KL 散度 $\text{KL}(\pi_\star \| \pi_\theta)$。**

#### RL 的反向 KL

从标准的 KL 正则化 RL 目标出发：

$$
\max_\pi \; \mathcal{J}_\text{RL}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \right] - \beta \cdot \text{KL}\!\left(\pi(\cdot \mid x) \| \pi_\text{ref}(\cdot \mid x)\right)
$$ {#eq:rl_objective_retaining}

提出 $-\beta$ 把最大化转成最小化：

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)} - \frac{1}{\beta} r(x, y) \right]
$$ {#eq:rl_min_form}

引入配分函数 $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)$ 把“奖励倾斜后的参考分布”归一化成合法分布，并加减 $\log Z(x)$，内层期望就变成一个 KL 散度：

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D}} \left[ \text{KL}\!\left(\pi(\cdot \mid x) \;\middle\|\; \frac{1}{Z(x)} \pi_\text{ref}(\cdot \mid x) \exp\!\left(\tfrac{1}{\beta} r(x,y)\right) \right) - \log Z(x) \right]
$$ {#eq:rl_kl_form}

由于 $\log Z(x)$ 不依赖 $\pi$，而 KL 散度非负且当且仅当两个分布相同时为零，所以当 $\pi$ 等于“奖励倾斜后的分布”时，KL 取到最小值 0。
因此奖励 $r(x,y)$ 下的最优策略是：

$$
\pi_\star(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)
$$ {#eq:optimal_policy_retaining}

现在可以直接展示与反向 KL 的联系。展开 $\text{KL}(\pi_\theta \| \pi_\star)$ 并代入 $\log \pi_\star(y \mid x) = \log \pi_\text{ref}(y \mid x) - \log Z(x) + \frac{1}{\beta} r(x, y)$：

$$
\begin{aligned}
\text{KL}(\pi_\theta \| \pi_\star) &= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\star(y \mid x) \right] \\
&= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\text{ref}(y \mid x) + \log Z(x) - \frac{1}{\beta} r(x, y) \right] \\
&= - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) + \underbrace{\log Z(x)}_\text{const} \\
&\propto - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) \\
&= -\frac{1}{\beta} \mathcal{J}_\text{RL}(\theta)
\end{aligned}
$$

等价地，**最大化 RL 目标 $\mathcal{J}_\text{RL}(\theta)$ 就是最小化反向 KL 散度 $\text{KL}(\pi_\theta \| \pi_\star)$。**

这个推导表明 SFT 与 RL 优化的是根本不同的目标：**SFT 最小化前向 KL，RL 最小化反向 KL。**

![前向 KL（SFT）与反向 KL（RL）下的遗忘动态。“旧”模态代表先验知识，“新”模态代表目标任务。前向 KL 把策略拉伸去覆盖目标，并把质量从旧模态抽走（右上）；而反向 KL 把新模态移向目标，却不扰动旧模态（右下）。出自 Chen 等 2025，经作者许可。](images/retaining_by_doing_mode_intuition.png){#fig:retaining-mode-intuition}

KL 散度的两个方向会诱导不同的优化压力。

前向 KL 在“目标分布有质量而模型没有”的地方惩罚模型，这倾向于鼓励**覆盖模态**（mode covering）——模型把概率铺开来覆盖目标的所有主要模态。
原因在于：前向 KL 的期望是在 $\pi_\star$ 下取的，所以当模型未能给目标有质量的区域分配概率时，惩罚很重。

反向 KL 只在模型**实际放置质量**的区域惩罚它，这倾向于鼓励**寻找模态**（mode seeking）：模型可以集中在一个高概率模态上而忽略其他模态。
这里期望是在 $\pi_\theta$（模型自己的分布）下取的，所以即使 $\pi_\star$ 在某处有可观质量，只要 $\pi_\theta(y \mid x) \approx 0$，那里对损失的贡献也很小。
同时，它会在模型把质量放到目标没有质量的地方时惩罚模型。

有了这个区分，我们可能会天真地预期 SFT 比 RL **遗忘更少**：覆盖模态的前向 KL 应当在目标的所有模态间维持质量、保住旧知识；而寻找模态的反向 KL 可能坍缩到单个高奖励模态、抛弃其他。
然而事实相反。
这个直觉假设了单模态策略，但预训练 LLM 包含多个模态——而对多模态分布，动态会翻转。

设想一个有两个模态的策略：一个代表先验知识的“旧”模态，一个对应目标任务的“新”模态（@fig:retaining-mode-intuition）。
前向 KL（SFT）试图同时覆盖目标分布的两个模态，这就推动策略去拉伸、把概率质量**从旧模态重新分配出去**，扰乱其形状、造成遗忘。
而反向 KL（RL）只需要把质量放到某个高奖励区域，因此它可以把“它采样到的”新模态移向目标，而完全不碰旧模态，从而让先验知识保持完整。

RL 的寻找模态行为——反向 KL 的一个结构性性质——保住了模型先验知识的广度，并带来更好的泛化。

总结：

- **SFT（前向 KL）**：$\text{KL}(\pi_\star \| \pi_\theta)$——样本来自目标 $\pi_\star$，一个固定的人类撰写补全数据集。对每个示例，我们问：我们的模型 $\pi_\theta$ 给它分配了多少概率？模型从不生成任何东西，它学的是模仿。这种覆盖模态的压力迫使策略大范围重新分配质量，可能扰乱先验知识。

- **RL（反向 KL）**：$\text{KL}(\pi_\theta \| \pi_\star)$——样本来自我们自己的策略 $\pi_\theta$。对模型生成的每个补全，我们问：它离奖励最优策略 $\pi_\star$ 有多近？因为模型只在自己的生成上训练，更新保持在它已经放置概率质量的地方——奖励信号告诉它该强化其中哪些生成，把概率移向 $\pi_\star$，而不扰动分布的其他部分。

### RL 剃刀：为什么在线 RL 遗忘更少

上一节表明同策略采样驱动了 RL 对遗忘的抵抗，并把机制追溯到前向/反向 KL 的动态。
对任何给定任务，都存在许多性能很高的不同策略。
Shenfeld 等 2026 [@shenfeld2026rls] 对 RL 的泛化提供了互补视角，提出**RL 剃刀**（RL's Razor）论题，其假说如下：

> 在一个新任务的众多高奖励解之中，像 RL 这样的同策略方法，内在地偏向那些在 KL 散度上更接近原始策略的解。

![偏向 KL 最小的解会减少遗忘。（左）在解决新任务的策略中，RL 收敛到与基座模型 KL 最接近的那些。（右）在相同的新任务性能下，这种 KL 偏置带来更高的旧任务保持率——相比 SFT。出自 Shenfeld、Pari 与 Agrawal 2026。许可 CC-BY。](images/rl_razor_motivation.png){#fig:rl-razor-motivation}


作者发现，对过去任务的遗忘，与微调后策略偏离初始模型的程度（以 KL 散度衡量）**直接成正比**：

$$
\text{Forgetting} \approx f\!\left(\mathbb{E}_{x \sim \tau}\!\left[\text{KL}\!\left(\pi_0(\cdot \mid x) \| \pi(\cdot \mid x)\right)\right]\right)
$$ {#eq:rl_razor_forgetting}


在 RL 与 SFT 的若干训练变体中，作者实证表明遗忘与“训练后策略同初始策略之间的 KL 散度”强相关（$R^2 = 0.96$）——而且**是用新任务数据测得的**。
这令人意外，因为该 KL 是在*新任务的*输入分布上测的，而不是在旧任务的留出数据上；但它仍能预测过去任务上的性能下降。
实践中，这给了我们一件有力的工具：直接用基座策略与训练后策略之间的漂移来估计遗忘——在新专精数据上测 KL 距离即可。

为弄清是什么驱动了 RL 策略中更小的 KL 偏移，作者沿两条轴分解了 RL 与 SFT 的差异——同策略数据 vs 离线数据，以及目标是否包含**负梯度**（RL 中当样本得分低于奖励基线时存在，SFT 中不存在，因为 SFT 只强化正确示范）来把概率推离错误输出。
值得注意的是：他们发现“同策略 vs 离线数据”完全解释了泛化性能的差异，而负梯度没有可辨别的影响。

直观上，同策略方法采样的输出，是模型本身已经赋予不可忽略概率的那些，因此每次更新都被约束在接近当前分布的地方。
另一方面，SFT 在一个固定的外部分布上训练，该分布可能与模型当前产出相距任意远，而每个梯度步都朝那个遥远目标拉——不论模型自己的信念如何。
