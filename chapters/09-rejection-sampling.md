<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
  代码块保留原文。
-->
---
prev-chapter: "直接对齐算法"
prev-url: "08-direct-alignment"
page-title: 拒绝采样
search-title: "第 9 章：拒绝采样"
meta-description: "用奖励或偏好信号改进后训练语言模型的拒绝采样与 best-of-n 方法。"
next-chapter: "偏好的本质"
next-url: "10-preferences"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "第 2 讲：IFT、奖励建模与拒绝采样（第 4、5、9 章）"
---

# 拒绝采样

拒绝采样（Rejection Sampling，RS）是偏好微调中**用得最广、记录得最少**的方法之一。
许多著名的 RLHF 论文都把它作为训练流程的核心组件，却不存在公认的实现，也没有解释它为何如此有效。
拒绝采样可以应用在训练流程的多个节点上——指令微调之后、基于 RL 的优化之后，甚至 RLVR 之后——这让它成为一个用途广泛、却难以归位的工具。
再加上它文档化程度低，这正是它被放在核心优化方法最后的原因。

拒绝采样的做法是：整理出一批新的候选补全，用训练好的奖励模型筛选，然后**只用最好的那些补全**对原模型做微调（损失函数与指令微调相同）。

这个名字来自计算统计学 [@gilks1992adaptive]：当你想从某个复杂分布中采样，却没有直接可行的方法时，就从另一个更容易建模的分布中采样，并用一个启发式规则检查该样本是否可用。
在语言模型上，目标分布是“对提示的高质量补全”，筛选器是奖励模型，采样分布则是当前模型。

WebGPT [@nakano2021webgpt]、Anthropic 的 Helpful and Harmless 智能体 [@bai2022training]、OpenAI 那篇关于过程奖励模型的著名论文 [@lightman2023let]、Llama 2 Chat 模型 [@touvron2023llama] 以及其他奠基性工作，都使用了这一基线做法；更新的工作则把它直接形式化（例如把拒绝采样应用到多模态对齐的 RAFT [@dong2023raft]，以及给出了“拒绝采样与其他偏好学习目标如何关联”之原则性综述的统计拒绝采样优化 RSO [@liu2023statistical]）。

*本章通篇用 $x$ 表示提示、$y$ 表示补全。这一记号在语言模型文献中很常见——那些方法作用于完整的“提示—补全”对，而非单个 token。*

## 训练流程，一步一步来

拒绝采样整体上分为几个阶段。

0. **选择提示与奖励模型：** 首先，你要相对于其他训练阶段来选择本阶段要训练的提示。最简单的方法是复用第一个 SFT/IFT 阶段的全部提示，但这可能导致一些过拟合。在做拒绝采样之前，你还必须已经训练好一个奖励模型（更多信息见第 5 章）。
1. **从起始检查点生成补全：** 接着，用你想优化的模型对选定的提示生成补全。这一步可能涉及调整许多设置，例如采样温度、top-p、最大序列长度、每个提示的补全数等。
2. **用奖励模型选出最好的补全**：所有补全由奖励模型排序。这一步也可能包含去重，即每个提示只保留一个补全；不过许多这类设计选择最终都要靠经验性的消融实验来定。
3. **在最好的补全上做 SFT：** 最后，用选出的补全对起始检查点做指令微调，拒绝采样就完成了。

拒绝采样流程的可视化总览见 @fig:rs-overview。

![拒绝采样总览。](images/rejection-sampling.png){#fig:rs-overview}

关于到底该用哪些提示、如何选奖励模型、如何安排拒绝采样的顺序等具体细节，文献中记录得并不充分。
本章给出方法概览，进一步的实验留给读者。

### 生成补全

要为每个提示生成一组多个候选补全，我们先把 $M$ 个提示的集合定义为一个向量：

$$X = [x_1, x_2, ..., x_M]$$ {#eq:rs_prompt_vector}

这些提示可以来自许多来源，但最常见的是来自指令训练集。

对每个提示 $x_i$，我们生成 $N$ 个补全。这可以表示为一个矩阵：

$$Y = \begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,N} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
y_{M,1} & y_{M,2} & \cdots & y_{M,N}
\end{bmatrix}$$ {#eq:rs_completion_matrix}

其中 $y_{i,j}$ 表示第 $i$ 个提示的第 $j$ 个补全。
每一行 $i$ 对应单个提示 $x_i$，包含它的 $N$ 个候选补全；每一列 $j$ 对应所有提示在“第 $j$ 次采样”上的补全。

### 给补全打分

现在，把所有这些“提示—补全”对送入奖励模型，得到一个奖励矩阵。
我们把奖励表示为矩阵 $R$：

$$R = \begin{bmatrix}
r_{1,1} & r_{1,2} & \cdots & r_{1,N} \\
r_{2,1} & r_{2,2} & \cdots & r_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
r_{M,1} & r_{M,2} & \cdots & r_{M,N}
\end{bmatrix}$$ {#eq:rs_reward_matrix}

每个奖励 $r_{i,j}$ 都由补全 $y_{i,j}$ 与其对应提示 $x_i$ 经过奖励模型 $\mathcal{R}$ 计算得到：

$$r_{i,j} = \mathcal{R}(y_{i,j} \mid x_i)$$ {#eq:rs_reward_computation}

选出“用来训练的最佳补全”有多种方法。

为了把“依据奖励矩阵挑出最佳补全”这个过程形式化，我们可以定义一个作用在奖励矩阵 $R$ 上的选择函数 $S$。

#### 每提示取最优

第一个可能的选择函数是取每个提示的最大奖励。

$$S(R) = \left[\arg\max_{j} r_{1,j}, \arg\max_{j} r_{2,j}, ..., \arg\max_{j} r_{M,j}\right]$$ {#eq:rs_selection_per_prompt}

这个函数 $S$ 返回一个索引向量，其中每个索引对应 $R$ 的每一行中奖励最大的那一列。
然后我们用这些索引挑出被选中的补全：

$$Y_{chosen} = [y_{1,S(R)_1}, y_{2,S(R)_2}, ..., y_{M,S(R)_M}]$$ {#eq:rs_chosen_completions}


#### 全体取前 K 对

另一种做法是从整个集合中选出前 $K$ 个“提示—补全”对。
先把奖励矩阵 $R$ 展平成一个向量：

$$R_{flat} = [r_{1,1}, r_{1,2}, ..., r_{1,N}, r_{2,1}, r_{2,2}, ..., r_{2,N}, ..., r_{M,1}, r_{M,2}, ..., r_{M,N}]$$ {#eq:rs_flattened_rewards}

这个 $R_{flat}$ 向量的长度为 $M \times N$，其中 $M$ 是提示数、$N$ 是每个提示的补全数。

现在我们可以定义一个选择函数 $S_K$，选出 $R_{flat}$ 中前 K 个最大值的索引：

$$S_K(R_{flat}) = \text{argsort}(R_{flat})[-K:]$$ {#eq:rs_topk_selection}

其中 $\text{argsort}$ 返回能把数组升序排列的索引，我们取最后 $K$ 个索引，就得到最大的 $K$ 个值。

要得到被选中的补全，需要把这些展平后的索引映射回原来的补全矩阵 $Y$。
把零基的展平索引 $k$ 映射到 $(i,j)$，可用 $i = \lfloor k / N \rfloor + 1$、$j = (k \bmod N) + 1$。

#### 选择示例

考虑下面这个情形：五个提示、四个补全。
我们演示两种依据奖励选择补全的方式。

$$R = \begin{bmatrix}
0.7 & 0.3 & 0.5 & 0.2 \\
0.4 & 0.8 & 0.6 & 0.5 \\
0.9 & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & 0.8 & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_matrix}

首先是**每提示取最优**。直观上，我们可以这样高亮奖励矩阵：

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & \textbf{0.6}
\end{bmatrix}$$ {#eq:rs_example_per_prompt}

用 argmax 方法，我们为每个提示选出最好的补全：

$$S(R) = \left[\arg\max_{j} r_{i,j} \text{ for } i \in [1,5]\right]$$ {#eq:rs_example_selection_formula}

$$S(R) = [1, 2, 1, 3, 4]$$ {#eq:rs_example_selection_result}

这意味着我们会选择：

- 提示 1：补全 1（奖励 0.7）
- 提示 2：补全 2（奖励 0.8）
- 提示 3：补全 1（奖励 0.9）
- 提示 4：补全 3（奖励 0.8）
- 提示 5：补全 4（奖励 0.6）

再看**全体取前 K**。
我们高亮出全体排名前五的“提示—补全”对。

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & \textbf{0.7} \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_top_overall}


先把奖励矩阵展平：

$$R_{flat} = [0.7, 0.3, 0.5, 0.2, 0.4, 0.8, 0.6, 0.5, 0.9, 0.3, 0.4, 0.7, 0.2, 0.5, 0.8, 0.6, 0.5, 0.4, 0.3, 0.6]$$ {#eq:rs_example_flattened}

然后选出五个最大值的索引：
$$S_5(R_{flat}) = [8, 5, 14, 0, 11]$$ {#eq:rs_example_topk_result}

把它们映射回原矩阵：

- 索引 8 → 提示 3，补全 1（奖励 0.9）
- 索引 5 → 提示 2，补全 2（奖励 0.8）
- 索引 14 → 提示 4，补全 3（奖励 0.8）
- 索引 0 → 提示 1，补全 1（奖励 0.7）
- 索引 11 → 提示 3，补全 4（奖励 0.7）

#### 实现示例

下面这段代码展示了这些选择方法可以如何实现。

```python
import numpy as np

x = np.random.randint(10, size=10)
print(f"{x=}")
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
print(f"{x_sorted=}")

# first way to recover the original array
i_rev = np.zeros(10, dtype=int)
i_rev[sorted_indices] = np.arange(10)
np.allclose(x, x_sorted[i_rev])

# second way to recover the original array
np.allclose(x, x_sorted[np.argsort(sorted_indices)])
```

### 微调

有了选出的补全，接下来就在模型的当前版本上做标准的指令微调。
更多细节见[指令微调那一章](https://rlhfbook.com/c/04-instruction-tuning)。

## 实现细节

执行这项训练的核心超参非常直观：

- **采样参数**：拒绝采样直接依赖模型给出的补全。常见设置包括使用大于 0 的温度，例如 0.7 到 1.0 之间，并对 top-p 或 top-k 采样等其他参数做调整。
- **每个提示的补全数**：成功的拒绝采样实现中，每个提示包含 10 到 30 个或更多补全。补全太少会让训练有偏和/或有噪声。
- **指令微调的细节**：拒绝采样中指令微调的确切训练细节并未公开。它们很可能使用了与模型初始指令微调阶段略不相同的设置。
- **异构模型生成**：一些拒绝采样的实现会纳入多个模型的生成，而不只是即将被训练的那个模型。这方面的最佳实践尚未确立。
- **奖励模型训练**：所用的奖励模型会极大影响最终结果。关于奖励模型训练的更多资料，见[相关章节](https://rlhfbook.com/c/05-reward-models)。

在做批量奖励模型推理时，可以按长度对 token 化后的补全排序，让同一批次内的长度相近。
这样就不必在那么多填充 token 上做推理，能以少量实现复杂度的代价提升吞吐。

## 相关方法：Best-of-N 采样

Best-of-N（BoN）是拒绝采样的近亲：沿用同样的“生成并打分”流程，但**不在**选出的补全上微调模型。
相反，BoN 是在推理时对某个固定提示（或一组提示）算出最好的那个补全；相关技术常被用于聊天模型的“Pro”档位——它们花额外算力来回答你的问题。

Best-of-N 采样常被当作相对于 RLHF 训练方法的基线。
要记住：BoN **不修改**底层模型，它是一种采样技术。
正因如此，把 BoN 采样与 PPO 这类在线训练方法做比较，在某些语境下仍然是有效的。
例如，你仍然可以测量运行 BoN 采样时相对于任何其他策略的 KL 距离。

这里我们要说明：当对单个提示做简单的 BoN 采样时，上面两种选择准则是等价的。

设 $R$ 为单个提示下 $N$ 个补全的奖励向量：

$$R = [r_1, r_2, ..., r_N]$$ {#eq:rewards_vector}

其中 $r_j$ 表示第 j 个补全的奖励。

用 argmax 方法，我们为这个提示选出最好的补全：

$$S(R) = \arg\max_{j \in [1,N]} r_j$$ {#eq:selection_function}

用 top-K 方法并取 $K=1$，就退化为同一个方法——这也是常见做法。

## 建议的实验

`code/rejection_sampling/` 中的配套实现跑通了一条完整的 GSM8K 拒绝采样流水线：生成 rollout、用奖励模型打分、选出训练子集、微调、评估精确匹配率。
四份配置被安排成配对的“处理组/对照组”，读者可以据此追问：**奖励模型究竟有没有起作用？**

1. **先构建一次 rollout 缓存。**

   ```bash
   cd code/
   uv run python -m rejection_sampling.preprocess \
       --config rejection_sampling/configs/top_per_prompt.yaml
   ```

   这会为共享的 GSM8K 切片生成并打分补全。
   只要生成与打分设置不变，后续的训练配置都会复用这份缓存。

2. **把奖励选择与随机对照做比较。**

   ```bash
   cd code/
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/top_per_prompt.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/random_per_prompt.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/top_k_overall.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/random_k_overall.yaml
   ```

   按配对读结果：`top_per_prompt` 对比 `random_per_prompt`，`top_k_overall` 对比 `random_k_overall`。
   如果奖励选出的那一组打不过它的随机基线，说明奖励模型或采样得到的补全没能在那个切片上提供有用的信号。

3. **改变“奖励模型有多少可挑”的程度。**
   复制一份配置，修改 `num_completions_per_prompt`、`temperature`、`top_p` 和 `selection.top_k`。
   更多补全可以改善“可选样本中的最好者”，但前提是奖励模型能区分好答案与坏答案。

4. **换一个更小的策略模型。**
   把 `model_name` 设为更小的、兼容的指令模型，降低 `max_train_samples`，重跑同样的配对实验。
   这样实验更便宜，也能凸显：拒绝采样究竟是在**挽救**弱生成，还是仅仅在**已经不错**的生成中做挑选。
