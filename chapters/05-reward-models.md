<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
  代码块与英文提示词保留原文。
-->
---
prev-chapter: "指令微调"
prev-url: "04-instruction-tuning"
page-title: 奖励建模
search-title: "第 5 章：奖励建模"
meta-description: "如何从偏好数据训练奖励模型，以及它如何在后训练流水线中充当 RLHF 的学习目标。"
next-chapter: "强化学习"
next-url: "06-policy-gradients"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "第 2 讲：IFT、奖励建模与拒绝采样（第 4、5、9 章）"
---

# 奖励建模

奖励模型是现代 RLHF 方法的核心——复杂的人类偏好正是在这里被学习到的。
正是它们让我们能从难以描述的**困难信号**中学习。
它们把数据中的复杂特征压缩成一种可供下游训练使用的表示——这可以说是一种“魔法”，再次展现了现代深度学习的复杂能力。
这些模型充当核心优化的**代理目标**，后续章节会详细研究这一点。
如 @fig:rm-role-in-rlhf 所示，奖励模型扮演的角色类似于标准 RL 中的环境：为智能体提供学习信号；但与固定的环境不同，我们可以从人类偏好中把它**学出来**。

历史上，奖励模型在强化学习研究中一直被广泛用作环境奖励的代理 [@sutton2018reinforcement]。
而现代形式的奖励模型，最初是作为研究“价值对齐”问题的工具被提出的 [@leike2018scalable]。
这类模型通常接受某种输入，输出一个标量奖励值。
奖励可以有多种形式——在传统 RL 问题中，它试图逼近该问题真实的环境奖励；而在 RLHF 中我们会看到，奖励模型实际输出的是“某个输入属于高质量”的概率（也就是在成对偏好关系中属于被选中的那个答案）。
RLHF 中的奖励建模实践与逆向强化学习密切相关——后者的任务是：给定行为轨迹，反推出智能体的奖励函数 [@ng2000algorithms]；它也与深度强化学习的其他领域相关。
两者在高层次的问题陈述上相同，但在实现与关注点上完全不同，所以通常被视为两个独立的研究领域。

最常见的奖励模型通常被称为 **Bradley-Terry 奖励模型**，也是本章的主要焦点。它预测一段文本**接近训练比较中“更受偏好”的文本**的概率。
本节后面我们还会把它与结果奖励模型（ORM）、过程奖励模型（PRM）以及其他类型的奖励模型作比较。

*本章通篇用 $x$ 表示提示、$y$ 表示补全。这一记号在语言模型文献中很常见——那些方法作用于完整的“提示—补全”对，而不是单个 token。*

![RLHF 中的奖励模型扮演标准 RL 里“返回奖励的环境组件”这一角色。关键区别在于：在 RLHF 中，这个奖励函数由我们从人类偏好中控制和学习，而不是由环境固定。](images/rlhf-overview.png){#fig:rm-role-in-rlhf}

## 训练 Bradley-Terry 奖励模型

奖励模型的经典实现源自 Bradley-Terry 偏好模型 [@BradleyTerry]。
训练标准 RLHF 奖励模型有两种常见的表达方式——它们在数学上等价。
首先，Bradley-Terry 偏好模型定义了：在两个条目 $i$ 与 $j$ 的成对比较中，评判者偏好 $i$ 而非 $j$ 的概率：

$$P(i > j) = \frac{p_i}{p_i + p_j}.$$ {#eq:bradterry}

Bradley-Terry 模型假设每个条目有一个潜在的强度 $p_i > 0$，观察到的偏好是这些潜在强度的带噪反映。
常见的做法是把 Bradley-Terry 模型用无界分数重新参数化，即令 $p_i = e^{r_i}$，于是得到如下形式：

$$P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i-r_j).$$ {#eq:bradterry_unbounded}

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 logistic（sigmoid）函数，因此偏好概率只取决于分数差 $r_i - r_j$。
**只有分数的差值有意义**：给每个 $r_k$ 同时加上一个常数 $c$，$P(i > j)$ 不变。
这些形式是对人类偏好的一个有用近似，在 RLHF 中往往效果很好。

要训练奖励模型，我们必须构造一个满足上述关系的损失函数。
实践中，做法是把语言模型改造成一个输出标量分数的模型，通常是在模型最终隐藏状态之上接一个小型线性头，产出一个奖励值。
给定提示 $x$ 与两个采样得到的补全 $y_1$ 和 $y_2$，我们用奖励模型 $r_\theta$ 给两者打分，条件分数记作 $r_\theta(y_i \mid x)$。

奖励模型给“$y_1$ 优于 $y_2$”分配的概率为：

$$P(y_1 > y_2 \mid x) = \frac{\exp\left(r_\theta(y_1 \mid x)\right)}{\exp\left(r_\theta(y_1 \mid x)\right) + \exp\left(r_\theta(y_2 \mid x)\right)}.$$ {#eq:bradterryrm}

我们把受偏好的补全记作 $y_c$（chosen，被选中），被拒绝的补全记作 $y_r$（rejected）。

由此得到的损失，会促使奖励模型给人类偏好的补全打出比被拒绝补全更高的分数，并用 sigmoid 把分数差转换成概率。
@eq:bradterryrm 中的偏好似然是起点。我们先把分子分母同除 $\exp\left(r_\theta(y_c \mid x)\right)$，把它改写成 sigmoid 形式：

$$
\begin{aligned}
P(y_c > y_r \mid x)
&= \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right) + \exp\left(r_\theta(y_r \mid x)\right)} \\
&= \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)\left(1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}\right)} \\
&= \frac{1}{1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}} \\
&= \frac{1}{1 + \exp\left(-(r_\theta(y_c \mid x) - r_\theta(y_r \mid x))\right)} \\
&= \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right).
\end{aligned}
$$ {#eq:bradterryrm_sigmoid}

随后，奖励模型通过在偏好数据集 $D$ 上做最大似然估计来拟合，也就是最大化观测偏好的期望对数似然。由于对数函数单调，这等价于最小化期望负对数似然：

$$
\begin{aligned}
\theta^* &= \arg\max_\theta \mathbb{E}_{(x, y_c, y_r) \sim D}\left[ \log P(y_c > y_r \mid x) \right] \\
&= \arg\min_\theta \mathbb{E}_{(x, y_c, y_r) \sim D}\left[ -\log \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right) \right].
\end{aligned}
$$ {#eq:bradterryrm_deriv}

**在对数据集求平均之前先取对数**，正是负对数似然成为正确目标的原因：最大化期望概率 $\mathbb{E}[P]$ 与最大化期望对数概率 $\mathbb{E}[\log P]$ 并不是一回事。

单样本损失就是上面期望内部的 log-sigmoid 表达式，如 [@ouyang2022training] 等工作中那样：
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodeling1}

第二种是数学上等价的形式，用 softplus 函数 $\log(1+e^x)$ 表达，如 [@askell2021general] 等工作中那样：
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)$$ {#eq:rewardmodeling2}

只要令 $\Delta = r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x)$，并利用 $\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$，即可得到 $-\log\sigma(\Delta) = \log(1 + e^{-\Delta}) = \log\left(1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)}\right)$，可见两者等价。
这两种写法在 RLHF 文献中都出现过。

![训练偏好奖励模型需要成对的“被选中”与“被拒绝”补全。模型从序列级表示（通常是序列结束（EOS）token 的隐藏状态）为每个补全计算一个标量分数，对比式损失只依赖两者分数之差。](images/pref_rm_training.png){#fig:pref_rm_training data-dark-src=“images/pref_rm_training-dark.png”}

### 默认的奖励模型架构

奖励模型最常见的实现方式，是借助与 Transformers 中 `AutoModelForSequenceClassification` 类似的抽象：在语言模型之上接一个小型线性头，在训练或推理时为“提示—补全”对产出一个标量奖励分数。
推理时，模型输出的这个单一 logit，就是*该段文本被选中的相对似然*。

也存在其他实现选择，比如直接从最终嵌入上接一个线性层，但它们在开源工具中不太常见。

### 实现示例

实现奖励建模的损失相当简单。
实现上更多的挑战在于搭建专门的数据加载器和推理流水线。
假设已经有了正确的数据加载器——把被选中和被拒绝的提示与补全都 token 化——损失可以这样实现：
```python
import torch.nn as nn
# inputs_chosen / inputs_rejected include the prompt tokens x and the respective
# completion tokens (y_c or y_r) that the reward model scores jointly.
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

从更大的图景看，这通常是在一个因果语言模型（从左到右生成 token、每个 token 都以之前所有 token 为条件来预测的模型）之上，额外加一个头（并用上面的损失来学习它），把最终隐藏状态映射为输入的分数。
这段代码接受标准的 transformer 输入——`input_ids`（token 化后的文本）与 `attention_mask`（标记真实 token 与填充）——并在最后一个真实 token 处取出隐藏状态（模型对输入的内部表示），再通过一个线性层产出一个标量奖励。
该模型的结构如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryRewardModel(nn.Module):
    """
    Standard scalar reward model for Bradley-Terry preference learning.

    Usage (pairwise BT loss):
        rewards_chosen = model(**inputs_chosen)    # (batch,)
        rewards_rejected = model(**inputs_rejected)  # (batch,)
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    """
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def _sequence_rep(self, hidden, attention_mask):
        """
        Get a single vector per sequence to score.
        Default: last non-padding token (EOS token); if no mask, last token.
        hidden: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        """

        # Index of last non-pad token in each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]  # (batch, hidden_size)

    def forward(self, input_ids, attention_mask):
        """
        A forward pass designed to show inference structure of a standard reward model.
        To train one, this function will need to be modified to compute rewards from both
         chosen and rejected inputs, applying the loss above.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]

        # One scalar reward per sequence: (batch,)
        seq_repr = self._sequence_rep(hidden, attention_mask)
        rewards = self.head(seq_repr).squeeze(-1)

        return rewards
```

在本节以及后文中，奖励模型（乃至后训练的大部分）的实现复杂度，主要都在于正确地构造数据加载器和分布式学习系统。
另外注意：训练奖励模型时，最常见的做法是**只训练 1 个 epoch**，以避免过拟合。

## 结果奖励模型

<!-- 原文致谢：感谢东北大学研究生 Hangliang Ren 协助撰写本节（以及 PRM 部分），见 https://github.com/myhott163com/RLHF_ORM_PRM -->

语言模型及其他 AI 系统的*偏好微调*，绝大多数都是用前面讨论的 Bradley-Terry 模型完成的。
对于推理密集型任务，可以使用**结果奖励模型**（Outcome Reward Model，ORM）。
ORM 的训练数据构造方式与标准偏好微调类似。
这里我们有一个问题陈述或提示 $x$，以及两个补全 $y_1$ 和 $y_2$。
它使用的归纳偏置是：其中一个补全是该问题的正确答案、另一个是错误答案，于是得到 $(y_c,y_{ic})$。

在继续之前必须指出：结果奖励模型在后训练文献中是一个相对小众的领域，我们引用的关键论文在实现细节上也有微妙差异。
其核心思想是学习一个**逐 token 的信号**，指示该补全有多大可能最终给出正确答案；但历史上出现过不同的训练方法和架构。

所用模型的架构与标准奖励模型非常相似——在能输出单一 logit 的模型上接一个线性层（对 RM 而言）——但对 ORM 来说，随后使用的训练目标略有不同。
先来拆解原始的 GSM8K 论文（一个研究小学数学习题的流行基准）[@cobbe2021gsm8k]，它提出了后来成为 ORM 的思想，但当时还没有这个名字。我们从第 4.3 节的架构说起：

> 我们既可以训练验证器基于整个生成的解答做单次标量预测，也可以在解答中的每个 token 之后做一次标量预测。
> 默认情况下我们选择后者，即训练验证器在每个 token 之后做预测。

这里正是结果奖励模型的默认实现与 Bradley-Terry 模型分道扬镳之处——**它们在每个 token 上做预测**。作者评论说，逐 token 的信息可以“作为一个有用的辅助信号，鼓励模型评判整个解答过程中的推理”，而不只是预测最终结果（这与后来 ORM 这个名字给人的印象有点反直觉）。继续看附录 E：

> \[我们\] 用一个联合目标训练验证器：模型除了原本的语言建模目标之外，还要学会把某个模型补全标注为正确或错误。
> 从架构上说，这意味着我们的验证器就是语言模型，只是带一个小型标量头，在**逐 token 的基础**上输出预测。
> 我们把这个标量头实现为单个偏置参数和单个增益参数，作用在语言模型最终 unembedding 层输出的 logits 上。

翻译成实现语言：这是在每个 token 上通过一个小头输出一个标量 logit，而不是像传统 RM 那样的分类头——整个序列只输出一个 logit。
另外，在这篇早期的 GSM8K 论文中，作者把 ORM 与下一词的语言建模损失**联合训练**；这一做法后来没有成为默认。

“结果奖励模型”（outcome-reward model）这个术语出现在 2022 年的一篇论文中，它比较了“结果监督的 RM（ORM）”与预测“截至目前推理质量”的过程奖励模型 [@uesato2022solving]——重要的是，这是 ORM 的第二种实现方式：把二元的 `correct` 或 `incorrect` 放进 LLM 的 tokenizer 词表里，作为**步骤级信号**，而不是学习一个在每 token 处预测正确性的独立标量头。

本书所遵循的经典实现来自论文 *Let's Verify Step by Step* [@lightman2023let]：其中结果奖励模型被训练成一个**逐 token 预测答案是否正确**的模型，用交叉熵损失。

形式上，逐 token 损失在每一个补全 token 上施加二元交叉熵，其中每个 token 关联的结果概率被训练去逼近该序列的结果标签：

$$\mathcal{L}_{\text{token}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}\left[\frac{1}{T}\sum_{t=1}^{T} \left( r\log p_\theta(s_t) + (1-r)\log\left(1-p_\theta(s_t)\right) \right)\right]$$ {#eq:orm_token_loss}

其中 $s$ 是长度为 $T$ 个 token 的补全，$r \in \{0,1\}$ 是二元标签（1 表示给定提示的正确答案，0 表示错误答案），而 $p_\theta(s_t) = \sigma(w_\theta(s_t))$ 是在 token $t$ 处由模型的标量 logit $w_\theta(s_t)$ 预测出的正确概率。

按照 [@lyu2025exploring]，ORM 还有一种更简单的形式：序列级交叉熵损失，模型之后用于逐 token 推理：

$$\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}\left[r\log \bar{p}_\theta(s) + (1-r)\log(1-\bar{p}_\theta(s))\right]$$ {#eq:orm_loss}

其中 $r \in \{0,1\}$ 是二元标签（1 表示给定提示的正确答案，0 表示错误答案），而 $\bar{p}_\theta(s) = \sigma\left(\frac{1}{T}\sum_{t=1}^{T} w_\theta(s_t)\right)$ 把逐 token logits 的平均值压缩成“整个补全正确”的单一概率——注意这**不是**逐 token 概率的平均，因为 sigmoid 是在汇聚之后才施加的。
在代码中，这个结果标签会被复制到每个补全 token 上，而提示 token 被掩蔽为 `-100`，不参与损失。

实现结果奖励模型（以及我们稍后会看到的过程奖励模型等其他类型）的做法是：依据补全是否为正确样本，**逐 token** 施加交叉熵损失。
这比标准 Bradley-Terry 奖励模型更接近语言建模损失——它不需要那种结构化的“选中—拒绝”配对关系。
在下面这个简化版 ORM 训练设置里，我们既不采样新 token，也不按下一词预测来训练 LLM；我们只是把一条固定的“提示—补全”序列喂给主干网络，训练 ORM 头去预测正确性标签。

模型结构可以写成：

```python
import torch.nn as nn
import torch.nn.functional as F

class OutcomeRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids contains a full prompt+completion sequence.
        labels is token-aligned: prompt tokens are -100, and each completion
         token repeats the sequence outcome label (1=correct, 0=incorrect).
        If labels=None, this is an inference-only forward pass and the loss is
         returned as None.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One scalar logit per token: (batch, seq_len)
        logits = self.head(hidden).squeeze(-1)

        # Inference-only forward pass: no loss is computed.
        if labels is None:
            return None, logits
        # Only compute loss on completion tokens (labels 0 or 1)
        # Prompt tokens have labels = -100
        mask = labels != -100
        loss = None
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], labels[mask].float()
            )
        else:
            loss = logits.sum() * 0
        return loss, logits
```

损失的简化写法如下：

```python
# Feed the full prompt+completion sequence once; no token sampling happens here.
# Assume model already has: model.lm (backbone) + model.head
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits_per_token = model.head(hidden).squeeze(-1)  # (batch, seq_len)
# This will sometimes be compressed as model.forward() in other implementations

# Binary labels: 1=correct, 0=incorrect (prompt tokens masked as -100)
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(
    logits_per_token[mask], labels[mask].float()
)
```

这里重要的直觉是：ORM 会在序列中的**每个 token** 处输出一个正确性概率（而判断依据只有最终答案——推理过程中的错误并不会被 ORM 的训练过程捕捉）。
这可能是一个带噪的过程，因为更新和损失会依据结果与注意力映射逐 token 地传播。

![推理时，结果奖励模型在补全 token 上输出逐 token 的正确性概率。提示 token 不参与打分，而补全上的概率可以聚合成回复级分数，用于验证、筛选或重排序。](images/orm_inference.png){#fig:orm_inference data-dark-src=“images/orm_inference-dark.png”}

![训练结果奖励模型使用来自验证器或数据集的离线标签（例如正确补全全部标为 1）。每个补全 token 都以二元交叉熵针对该结果标签训练，逐 token 概率再聚合成最终分数，用于验证、筛选或重排序。](images/orm_training.png){#fig:orm_training data-dark-src=“images/orm_training-dark.png”}

这类模型一直被使用，但在开源 RLHF 工具中支持较少。
例如，奠基性工作 *Let's Verify Step by Step* [@lightman2023let] 用的就是同一类 ORM，只是去掉了 Cobbe 等 2021 年工作中那部分语言建模预测损失。
于是，最终损失就是在每个 token 上的交叉熵损失，预测最终答案是否正确。

由于缺乏工具支持，“结果奖励模型”（ORM）这个术语被以多种方式使用。
一些文献（例如 [@lyu2025exploring]）继续受 Cobbe 等 2021 年原始定义的启发；另一些则把它更宽泛地用于任何“被训练来预测补全是否正确”的验证器。


## 过程奖励模型

过程奖励模型（Process Reward Model，PRM），最初叫“过程监督奖励模型”，是一类被训练来在思维链推理过程的**每一步**都输出分数的奖励模型。
它们不同于只在 EOS token 处输出一个分数的标准 RM，也不同于在每个 token 都输出分数的 ORM。
过程奖励模型需要在每个推理步骤的末尾获得监督，然后以类似方式训练——步骤内的 token 被训练去逼近相应的目标：对 PRM 来说目标是该步骤，对 ORM 来说目标是整个回复。

按照 [@lightman2023let]，二元标注的 PRM 通常用逐步骤交叉熵损失来优化：

$$\mathcal{L}_{\text{PRM}}(\theta) = - \mathbb{E}_{(x, s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} y_{s_i} \log r_\theta(s_i \mid x, s_{< i}) + (1 - y_{s_i}) \log \left(1 - r_\theta(s_i \mid x, s_{< i})\right) \right] $$ {#eq:prm_loss}

其中 $s$ 是一条采样得到的、带 $K$ 个标注步骤的思维链，$y_{s_i} \in \{0,1\}$ 表示第 $i$ 步是否正确，而 $r_\theta(s_i \mid x, s_{< i})$ 是 PRM 在给定原始提示 $x$ 与之前所有步骤 $s_{< i}$ 的条件下，预测步骤 $s_i$ 有效的概率。

下面是一个示例，展示这种逐步骤标签如何在训练器里被打包，来自 Hugging Face 的 TRL（Transformer Reinforcement Learning）[@vonwerra2022trl]：

```python
# Get the ID of the separator token and add it to the completions
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# Create the label 
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

传统上，PRM 用一个语言建模头来训练，只在推理步骤末尾输出一个 token——例如对应该步骤结束的那个双换行符或其他特殊 token 的位置。
这些预测通常取 -1 表示错误、0 表示中性、1 表示正确。
这些标签并不必然与“模型是否走在正确路径上”绑定，而是与该步骤本身是否正确绑定。

![过程奖励模型只在步骤边界（例如换行 token）提供监督。每个步骤获得一个三分类标签：正确（+1）、中性（0）或错误（-1）。训练时其他所有 token 都被掩蔽。](images/prm_training_inference.png){#fig:prm_training_inference data-dark-src=“images/prm_training_inference-dark.png”}

一个 PRM 的构造示例如下。

```python
import torch.nn as nn
import torch.nn.functional as F

class ProcessRewardModel(nn.Module):
    def __init__(self, base_lm, num_classes=3):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        The inputs are tokenized prompts and completions, where the end of a
         "reasoning step" is denoted by a designated separator token such as a
         newline or other special marker rather than batch padding.
        labels will be a list of labels, True, False, and Neutral (3 labels) which
         will be predicted by the model.
        If labels=None, this is an inference-only forward pass and the loss is
         returned as None.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One logit vector per token: (batch, seq_len, num_classes)
        logits = self.head(hidden)

        # Inference-only forward pass: no loss is computed.
        if labels is None:
            return None, logits
        # Only compute loss at step boundaries (where labels != -100)
        # Labels map: -1 -> 0, 0 -> 1, 1 -> 2 (class indices)
        mask = labels != -100
        loss = None
        if mask.any():
            loss = F.cross_entropy(
                logits[mask], labels[mask]
            )
        else:
            loss = logits.sum() * 0
        return loss, logits
```

核心损失函数与结果奖励模型看起来非常相似，只是标签施加在**不同的间隔**上。
```python
# Assume model outputs 3-class logits per token
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits = model.head(hidden)  # (batch, seq_len, 3)

# 3-class labels at step boundaries only: 0=-1, 1=0, 2=1 (others masked as -100)
mask = labels != -100
loss = F.cross_entropy(logits[mask], labels[mask])
```

## 比较各类奖励模型（以及价值函数）

前面介绍的各种奖励模型类型，展现了 RLHF 及其他后训练方法中衡量“质量”的多种光谱式做法。
下表汇总了这些模型各自预测什么、以及如何训练。

| 模型类别 | 预测什么 | 如何训练 | 语言模型结构 |
|---|---|---|---|
| **奖励模型** | 序列级质量分 $r_\theta(x, y)$ | 对同一提示下不同补全之间的成对（或 N 元）比较做对比式损失 | 在 EOS/最后 token 隐藏状态上接线性头 |
| **结果奖励模型** | 逐 token 的“答案正确”概率 | 带标签的结果（例如可验证领域上的成功/失败）；每个样本独立标注，不需要同一提示下的配对比较 | 逐 token 的二元交叉熵头；标签重复该结果标签 |
| **过程奖励模型** | 推理步骤末尾处对中间步骤的奖励或分数 | 用中间反馈或步骤级标注训练（在推理步骤内逐 token 训练） | 逐 token 预测步骤正确性（-1、0、1）的头 |
| **价值函数** | 给定当前状态的期望回报 | 通过回归到序列中的每个点来训练 | 带逐 token 输出的标量回归头 |
Table: 各类奖励模型的比较。 {#tbl:rm_compare}

关于上表这些区分，有几点需要说明——这些模型类型之间的边界并不总是那么清晰：

- 在偏好微调和推理训练中，价值函数的折扣因子常常取 1，这让价值函数更接近结果奖励模型，但训练损失不同。
- 过程奖励模型也可以通过从中间状态做 rollout、收集结果数据来获得监督。这混合了多种思路；但如果*损失*使用的是逐推理步骤的标签，那最好还是称之为 PRM。

**如果用“正确/错误”配对去训练 Bradley-Terry 成对模型，会怎样？**
关于结果奖励模型的许多混淆，来自一小部分文献：它们在由答案正确性导出的成对数据上训练奖励模型。
在这个设定里，你把“对某个问题的正确答案”设为被选中回复，把“*同一个问题的*错误答案”设为被拒绝回复。
这严格来说并不是 ORM，而是仍然直接用序列级的对比式损失训练的。
它严格来说仍是 Bradley-Terry 模型，属于我们讲的第一类模型。

**ORM 与价值函数的区别。**
ORM 和价值函数看起来可能相似，因为两者都通过同样的头部架构产出逐 token 输出；但它们在*预测什么*和*目标从哪来*上不同：

- **ORM** 在每个 token 处预测该补全**是否最终会给出正确答案**。目标来自*离线标签*（验证器或数据集把序列标为正确或错误），并被广播到每个中间 token 上用于训练。
- **价值函数** 预测**剩余**期望回报：$V(s_t) = \mathbb{E}\left[\sum_{k \geq t} \gamma^{k-t} r_k \mid s_t\right]$。目标通常是在当前策略 $\pi_\theta$ 下*同策略 rollout 计算*出来的，并随策略变化而变化（严格来说价值函数也可以是异策略的，但在语言模型的研究中这一点尚未确立）。

如果你定义稠密的逐 token 奖励 $r_t = \mathbb{1}[\text{token 正确}]$ 并取 $\gamma = 1$，那么 ORM 学的是 $r_t$（或 $p(r_t = 1)$），而价值头学的是剩余求和 $\sum_{k \geq t} r_k$。
它们可以共享同一个基座模型和相同的头部维度，但*语义与监督流水线*不同：ORM 从固定标签离线训练，而价值函数在同策略数据上训练，并被用来为策略梯度计算优势 $A_t = \hat{R}_t - V_t$。

### 各类奖励模型的推理方式

训练完成之后，这些模型在推理时对数据的处理方式不同，以应对 RM 被用于的一系列任务。

**Bradley-Terry RM（偏好模型）：**

- *输入：* 提示 $x$ + 候选补全 $y$
- *输出：* 由 EOS/最后 token 隐藏状态经线性层得到的单个标量 $r_\theta(x, y)$
- *用途：* 对 $k$ 个补全重排序、取 top-1（best-of-N 采样）；或为 RLHF 提供终止奖励
- *聚合：* 标量输出无需聚合

**结果 RM：**

- *输入：* 提示 $x$ + 补全 $y$
- *输出：* 补全 token 上的逐 token 概率 $p_t \approx P(\text{最终答案正确} \mid y_{\leq t})$
- *用途：* 给已完成的候选打分；用均值、最小值（尾部风险）或乘积 $\prod_t p_t$（等价于对数概率求和 $\sum_t \log p_t$）来聚合
- *聚合选择：* 平均正确性、$p_t$ 的最小值、最后 $m$ 个 token 的平均，或在任一 $p_t < \tau$ 时标记告警

**过程 RM：**

- *输入：* 提示 $x$ + 带步骤边界的推理轨迹
- *输出：* 步骤边界处的分数（例如正确/中性/错误的三分类 logits）
- *用途：* 给已完成的思维链打分；或通过剪掉低分分支来引导搜索/解码
- *聚合：* 在**步骤**（而非 token）上聚合——步骤分均值、最小值（快速失败），或偏向靠后步骤的加权和

**价值函数：**

- *输入：* 提示 $x$ + 当前前缀 $y_{\leq t}$（一个状态）
- *输出：* 补全中每个 token 位置处的 $V_t$（从状态 $t$ 出发的期望剩余回报）
- *用途：* 在 RL 训练中计算逐 token 优势 $A_t = \hat{R}_t - V_t$；每一步的值充作基线
- *聚合：* 通常取最后一个生成 token 处的 $V$；其解释与“正确性概率”不同

总结一下理解这些模型的方式：

- **RM：**“整个回答有多好？” → 一个标量值
- **ORM：**“这个回答最终会正确吗？” → 逐 token 的结果预测（作为中间质量的代理）
- **PRM：**“推理步骤站得住脚吗？” → 逐步分数
- **价值：**“从这里出发还剩多少奖励？” → RL 优势的基线

## 其他奖励模型变体

奖励建模是 RLHF 中相对探索不足的领域。
传统的 Bradley-Terry 奖励建模损失被许多流行工作修改过，但这些修改并未固化成单一的最佳实践。

### 偏好间隔损失

当标注者给出的是李克特量表（一种带有序数类别、指示偏好强度的评分表，例如 1–5 分）上的分数或排序时，这些关系量的**幅度**可以用在训练里。
最常见的做法是沿偏好方向把数据二值化，把相对评分或排序强度所携带的混合信息，压缩成仅仅是“被选中”与“被拒绝”的补全。
那些额外信息（例如偏好幅度）曾被用来改进模型训练，但没有收敛为标准做法。
Llama 2 提出用两个数据点之间的间隔 $m(y_c, y_r)$ 来区分偏好强度：

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)$$ {#eq:rewardmodelingmargin}

例如，每个补全常常会按质量被打 1 到 5 分。
如果被选中样本得 5 分、被拒绝样本得 2 分，那么间隔 $m(y_c, y_r)= 5 - 2 = 3$。
也可以探索其他计算间隔的函数。

注意在 Llama 3 中，这个间隔项被去掉了——团队观察到在扩大规模后改进变得微不足道。

### 平衡每个提示下的多次比较

InstructGPT 研究了用 $K = 4$ 到 $9$ 个补全对每个提示做排序的影响，从每个提示产生 $\binom{K}{2}$ 个成对比较 [@ouyang2022training]。
由于这些比较高度相关（它们共享同一个提示），把它们朴素地打散进数据集会导致奖励模型过拟合。
为解决这一点，他们按“每个提示下的每次比较”对损失更新加权——如果不重新加权，拥有更多补全的提示仅仅因为产生了更多配对，就会贡献更多总损失。
实践中，来自同一个提示的全部 $\binom{K}{2}$ 个比较通常被放进同一个训练批次并一起平均，因此每个提示只贡献一次分组更新，而不会分散到许多不同批次里。
这降低了对单个提示的过拟合，也避免了采样补全更多的提示主导损失。
损失函数变为：

$$\mathcal{L}(\theta) = - \frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_c, y_r)\sim D} \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodelinginstructgpt}


### K 元损失函数

还有许多其他形式可以为 RLHF 构造合适的人类偏好模型。
一个例子出现在早期的流行 RLHF 模型 Starling 7B 和 34B [@zhu2024starling] 中：基于 Plackett-Luce 模型的 K 元损失函数 [@liu2019learning]。

Zhu 等 2023 [@zhu2023principled] 把设定形式化如下。
给定一个提示（或称状态）$s^i$，从 $P(a_0,\cdots,a_{K-1}|s^i)$ 中采样 $K$ 个动作 $(a_0^i, a_1^i, \cdots, a_{K-1}^i)$。
然后标注者按偏好对这 $K$ 个动作排序，产生一个置换 $\sigma^i: [K] \mapsto [K]$，其中 $\sigma^i(0)$ 是最受偏好的动作。对所有 $K$ 个条目的完整排序，Plackett-Luce 概率为：

$$P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}$$ {#eq:kwise_rm}

当 $K = 2$ 时，它退化为成对比较的 Bradley-Terry（BT）模型。
无论如何，训练完成之后，这些模型在 RLHF 训练中的用法与其他奖励模型类似。


## 生成式奖励建模（即 LLM 作为评判者）

考虑到偏好数据的成本，一个大的研究领域出现了：用现有语言模型来评判人类偏好，或用于其他评估场景 [@zheng2023judging]。
核心思想是给语言模型一段“如何评判”的指令、一个提示，以及两个补全（就像交给人类标注者那样）。
下面是一个示例提示，来自聊天评估 MT-Bench 的奠基性工作之一 [@zheng2023judging]：

```text
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.
You should choose the assistant that follows the user's instructions and answers the user's question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.
Begin your evaluation by comparing the two responses and provide a short explanation.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names of the assistants.
Be as objective as possible.
After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
```

鉴于 LLM 作为评判者在评估中效果显著——它催生了 AlpacaEval [@dubois2024length]、Arena-Hard [@li2024crowdsourced]、WildBench [@lin2024wildbench] 等许多评估——不少人开始用 LLM 作为评判者来替代奖励模型，以生成并使用偏好数据。

围绕如何使用所谓“生成式奖励模型”，已经形成了一个完整的研究领域 [@mahan2024generative]
[@zhang2024generative] [@ankner2024critique]（包括*专门*训练来当好评判者的模型 [@kim2023prometheus]）；但在 RM 评估上，它们往往落后于现有的奖励模型——这说明奖励建模对当前的 RLHF 而言仍是一项重要技术。

提升 LLM 作为评判者这一流程鲁棒性的一个常用技巧，是使用采样温度 0 来降低评分的方差。

## 延伸阅读

奖励建模的学术文献在 2024 年确立了自己的位置。
奖励建模早期进展的大部分，集中在建立基准与识别行为模式上。
第一个 RM 基准 RewardBench 为测试奖励模型提供了通用基础设施 [@lambert2024rewardbench]。
此后，RM 评估已扩展到与通用后训练模型可用的评估类型相似：有些评估测试在具有已知真实答案的领域上的预测准确率 [@lambert2024rewardbench]，另一些则更接近“感觉”，用 LLM 作为评判者、或考察与其他基准的相关性 [@wen2024rethinking]。

新基准的例子包括：

- **纯文本（通用聊天/偏好）：** RMB [@zhou2024rmb]、RewardBench2 [@malik2025rewardbench]、Preference Proxy Evaluations [@frick2024evaluate]、RM-Bench [@liu2024rm]。
- **专门化的纯文本（数学等）：** 多语言奖励基准 M-RewardBench [@gureja2024m]、面向检索增强生成（RAG）的 RAG-RewardBench [@jin2024rag]、针对拼写错误的 ReWordBench [@wu2025rewordbench]、RewardMATH [@kim2024evaluating]、AceMath-RewardBench [@liu2024acemath]。
- **过程 RM：** PRM Bench [@song2025prmbench]、ProcessBench [@zheng2024processbench]，以及视觉基准 VisualProcessBench [@wang2025visualprm]、ViLBench [@tu2025vilbench]。
- **智能体 RM：** Agent-RewardBench [@men2025agentrewardbench]、CUARewardBench [@lin2025cuarewardbench]。
- **多模态：** MJ-Bench [@chen2024mj]、Multimodal RewardBench [@yasunaga2025multimodal]、VL RewardBench [@li2024vlrewardbench]、VLRMBench [@ruan2025vlrmbench]。

要了解奖励模型*训练*方面的进展，可以参考新的奖励模型训练方法：aspect 条件模型 [@wang2024interpretable]、高质量人类数据集 [@wang2024helpsteer2] [@wang2024helpsteer2p]、扩展实验 [@adler2024nemotron]、大量实验 [@touvron2023llama]，以及数据去偏 [@park2024offsetbias]。

## 建议的实验

配套代码仓库里有一些小型的奖励模型训练脚本，位于 `code/reward_models/`。
它们的定位是学习练习，而不是调好的参考配方。
先从干净的 `code/` 环境出发、执行 `uv sync`，然后一次跑一个实验。

1. **在 UltraFeedback 上训练一个 Bradley-Terry 偏好奖励模型。**
   运行：

   ```bash
   cd code/
   uv run python -m reward_models.train_preference_rm --config reward_models/configs/preference_rm.yaml
   ```

   观察演示输出与 W&B 日志中，被选中与被拒绝回复之间的奖励间隔是否在变大。
   然后在 yaml 配置里改变 `samples`、`lr` 和 `model_id`，看看信号在什么时候变得有噪声或不稳定。

2. **对比结果监督与过程监督。**
   分别运行 GSM8K 结果奖励模型与 PRM800K 过程奖励模型：

   ```bash
   cd code/
   uv run python -m reward_models.train_orm --config reward_models/configs/orm.yaml
   uv run python -m reward_models.train_prm --samples 500 --epochs 2
   ```

   对比训练之后两个模型能给出什么分数：ORM 应能区分正确的最终答案与错误的最终答案，而 PRM 应在中间的推理步骤上给出分数。
   这就是“序列级、结果级、过程级监督”这三者区别的实操版本。

3. **加一个小的留出集奖励模型评估。**
   一个有价值的贡献是为 `reward_models/` 写一个 50 到 200 条样本的评估，报告准确率或偏好对排序，且不需要完整跑一次训练。
   评估要足够小，以便在调超参时随时使用。
