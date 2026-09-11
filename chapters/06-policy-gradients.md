<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
  代码块保留原文。
-->
---
prev-chapter: "奖励建模"
prev-url: "05-reward-models"
page-title: 强化学习
search-title: "第 6 章：强化学习"
meta-description: "面向 RLHF 与 LLM 后训练的策略梯度方法，涵盖 PPO、REINFORCE、RLOO、GRPO 及实现细节。"
next-chapter: "推理与推理时扩展"
next-url: "07-reasoning"
lectures:
  - video: "https://www.youtube.com/watch?v=K_Sj_-1BUMM&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=4"
    label: "第 3 讲：理解用于 LLM 的策略梯度算法"
  - video: "https://www.youtube.com/watch?v=i-AIMpZHgeg&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=5"
    label: "第 4 讲：为 LLM 实现 RL 算法"
---

# 强化学习

在 RLHF 过程中，强化学习算法依据奖励模型给出的反馈，缓慢地更新模型权重。
策略——也就是被训练的那个模型——针对训练集中的提示生成补全，奖励模型给这些补全打分，然后强化学习优化器依据这些信息做梯度步（总览见 @fig:rlhf-overview）。
本章讲解这些算法背后的数学与权衡——它们用来从奖励模型给予同策略数据的信号中学习。
这些算法会运行许多个 epoch，往往在更大规模的提示集上跑数千到数百万个批次，每批之间都做梯度更新。

## 强化学习在 RLHF 中的角色

让 RLHF 在语言模型上流行起来的算法，是**策略梯度**类强化学习算法。
这类算法——例如近端策略优化（PPO）、组相对策略优化（GRPO）和 REINFORCE——使用最近生成的样本更新模型，而不是像 Deep Q-Network（DQN，AlphaGo 这类项目所用的方法）那样把分数存进经验回放缓冲区。
本节我们会介绍策略梯度算法的基础，以及它们如何在现代 RLHF 框架中使用。

从机器学习层面看，这一部分是 RLHF 流程中复杂度最高的主题。
不过，和大多数现代 AI 模型一样，其成败最大的决定因素是喂进这个流程的数据。

![RLHF 训练循环总览。数据集中的一个提示被送入已调优的策略，策略生成一个补全。奖励模型给这个补全打分；同时，被冻结的初始模型（通常是 RL 之前的指令微调模型）在同一段文本上计算对数概率，用于计算防止过度漂移的 KL 惩罚。合并后的奖励信号再驱动对策略参数的强化学习更新。](images/rlhf-overview.png){#fig:rlhf-overview}

RLHF 随着 ChatGPT 登场时，人们普遍知道它用的是 PPO 的一个变体，许多早期工作都建立在此之上。
后来，多个研究项目展示了 REINFORCE 风格算法的前景 [@ahmadian2024back] [@wang2024helpsteer2p]——它们被推崇的理由是比 PPO 更简单：不需要单独的价值模型（省显存，因而也省所需 GPU 数），优势估计也更简单（不需要广义优势估计 GAE，这是一种在策略梯度算法中用于降低方差的优势计算方法）。
此后又出现了更多算法，包括在推理任务上特别流行的组相对策略优化；但总体来说，这些算法大多可以针对具体任务来调。
本章覆盖策略梯度的核心设定，以及上面提到的三种算法——因为它们在建立经典 RLHF 文献中处于中心地位。

最简单地说，RLHF 的 RL 阶段需要两个模型：一个策略（被训练的模型），以及一个给它输出打分的奖励模型（上一章已讲）。
RL 之前的策略的一份副本充当**参考模型**，用于计算 KL 惩罚（这个模型是冻结的，即不会通过自动微分引擎接收梯度更新）。
本章讲到的最复杂算法 PPO，会增加第四个模型——一个学到的价值函数，用来估计动作中每个 token 有多好，它同样是一个在训练中更新的大语言模型。
本章各算法的主要区别在于：它们如何估计一个叫作**优势**（advantage）的量——衡量模型当前动作（补全）相对平均水平好多少；以及它们如何约束策略更新，使优化在数值上稳定。
这一 RLHF 过程（不含价值模型）的可视化总览见 @fig:rlhf-overview。

符号定义见问题设定那一章。

*本章使用强化学习文献中的 $(s, a)$ 记号，其中 $s$ 表示状态、$a$ 表示动作。在语言模型语境中，你更常看到的是 $(x, y)$，其中 $x$ 是提示、$y$ 是补全。$(s, a)$ 的框架更一般——这些算法本来就是为“在每个时间步采取动作”的序列决策问题设计的。不过，许多 RLHF 实现把整个补全当作单个动作，因此 $(x, y)$ 记号同样成立。*

***RL 速查表：** 本章全部核心 RL 损失函数的一页速查表见 [rlhfbook.com/rl-cheatsheet](https://rlhfbook.com/rl-cheatsheet)。*

## 策略梯度算法

本章的核心，是理解下面这个形状的方程。
这个方程计算的是我们正在训练的语言模型 $\pi_\theta$ 的梯度 $\Delta \theta$：

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$ {#eq:policy_gradient_intuition}

这里，方程由两个关键部分构成：
1. $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$——参数空间中**哪个方向**会让动作 $a_t$ 更可能发生。
2. $\Psi_t$——它**有多好**？一个给结果打分的标量。

把两者相乘，你就得到了策略梯度的更新量。
有些事很简单，比如 $\Psi_t > 0$ 会更新参数使 $a_t$ 更可能发生，$\Psi_t < 0$ 则使它的可能性降低。
策略梯度在计算的是：**哪些参数对某个动作有贡献**，以及我们是否应该让它将来更可能或更不可能发生。
本章余下的部分会深入探讨实现这件事的不同方式，以及让它对 LLM 奏效的具体技巧。

现在，我们把这件事再形式化一些。
强化学习算法的设计目标，是在由状态 $s \in \mathcal{S}$ 与动作 $a \in \mathcal{A}$ 组成的轨迹上，最大化未来的折扣奖励（更多记号见附录 A「定义」）。
智能体的目标——通常称为*回报*（return）——是从时刻 $t$ 起的折扣奖励之和（其中 $\gamma\in [0,1]$ 是一个让近期奖励更受重视的因子）：

$$G_t = r_t + \gamma r_{t+1} + \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k}.$$ {#eq:return_definition}

回报定义也可以递归地写成：
$$G_{t} = r_t + \gamma G_{t+1}.$$ {#eq:recursive_return}

这个回报是学习价值函数 $V(s)$ 的基础——后者是在给定当前状态下对**未来回报**的估计：

$$V(s) = \mathbb{E}\left[G_t \mid S_t = s \right].$$ {#eq:value_function}

所有策略梯度算法都在优化策略 $\pi_\theta(a\mid s)$ 以最大化期望回报；这个目标可以用诱导出的价值函数 $V^{\pi_\theta}(s)$ 来表达。

设 $d_0(s)$ 为初始状态分布。我们要最大化的回合式目标可以写成：
$$
J(\theta)
\;=\;
\sum_{s} d_0(s) V^{\pi_\theta}(s),
$$ {#eq:policy_objective}

在有限 MDP 中，这是对可能起始状态的求和；但实践中我们从不精确计算它。
我们改为从数据中估计：从当前策略采样 rollout。
在 RLHF 中，这通常意味着从数据集采样提示 $x_i$，并生成补全 $y_i \sim \pi_\theta(\cdot\mid x_i)$。
令 $R(x_i, y_i)$ 表示分配给该提示—补全对的标量序列级奖励；如果 $\tau_i$ 是对应的回合，这就是轨迹奖励 $R(\tau_i)$。
于是我们取一个经验平均，例如：

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} R(x_i, y_i),
$$ {#eq:empirical_batch_estimate}

或者，在带逐步奖励的 MDP 视角下：

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \sum_{t=0}^{T_i} \gamma^t r_{i,t}.
$$ {#eq:empirical_mdp_estimate}

实践中，面向语言模型的 RLHF 取 $\gamma = 1$（不折扣），因为优化的单位是整个补全、而不是单个 token——这一选择在本章后面「MDP 与赌博机」一节还会讨论。

策略梯度算法的核心，是计算“当前策略下有限期期望回报”的梯度。
有了这个期望回报 $J$，参数更新可以这样计算，其中 $\alpha$ 是学习率：

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$ {#eq:policy_update}

而核心的实现细节，就在于**如何计算这个梯度**。

### 推导策略梯度

令 $p_\theta(\tau)$ 表示由初始状态分布 $d_0$、策略 $\pi_\theta$ 和环境转移动力学共同诱导出的轨迹分布，其展开式见下面 @eq:trajectory_probability。
我们要最大化的 RL 目标还可以这样表述：
$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ R(\tau) \right],
$$ {#eq:policy_objective_expectation}

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是一条轨迹，$R(\tau) = \sum_{t=0}^\infty r_t$ 是这条轨迹的总奖励。或者，我们可以把这个期望写成对所有可能轨迹的积分：
$$
J(\theta) = \int_\tau p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_objective_integral}

注意，轨迹概率可以这样表达，其中 $\pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)$ 把策略概率与“从一个状态-动作对转移到下一个状态”的环境转移概率组合在一起：
$$
p_\theta (\tau) = d_0(s_0) \prod_{t=0}^\infty \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t),
$$ {#eq:trajectory_probability}

如果对目标函数（@eq:policy_objective_expectation）关于策略参数 $\theta$ 求梯度：
$$
\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_gradient_integral}

注意，我们可以用[对数求导技巧](https://andrewcharlesjones.github.io/journal/log-derivative.html)，把对积分的求导重写成一个期望：
$$
\begin{aligned}
\nabla_\theta \log p_\theta(\tau) &= \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} &\text{(from chain rule)} \\
\implies \nabla_\theta p_\theta(\tau) &= p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) &\text{(rearranging)}
\end{aligned}
$$ {#eq:log_chain_rule}

利用这个对数求导技巧：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau \\
&= \int_\tau p_\theta (\tau) R(\tau) \nabla_\theta \log p_\theta (\tau) d\tau \\
&= \mathbb{E}_{\tau \sim p_\theta} \left[ R(\tau) \nabla_\theta \log p_\theta (\tau) \right]
\end{aligned}
$$ {#eq:policy_gradient_expectation}

最后一步用的是轨迹分布 $p_\theta(\tau)$ 下期望的定义：对任意函数 $f$，有 $\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] = \int_\tau f(\tau)\,p_\theta(\tau)\,d\tau$（离散情形则是求和）。
把它写成期望很有用，因为我们可以用蒙特卡洛 rollout 来近似它，例如对由当前策略诱导的轨迹 $\tau_i \sim p_\theta$ 取 $\frac{1}{B}\sum_{i=1}^{B} f(\tau_i)$。

回到推导，展开轨迹的对数概率：

$$
\log p_\theta (\tau) = \log d_0(s_0) + \sum_{t=0}^\infty \log \pi_\theta(a_t|s_t) + \sum_{t=0}^\infty \log p(s_{t+1}|s_t, a_t)
$$ {#eq:trajectory_log_prob}

现在对它求梯度，我们得到：

- $\nabla_\theta \log d_0(s_0) = 0$（初始状态分布不依赖 $\theta$）
- $\nabla_\theta \log p(s_{t+1}|s_t, a_t) = 0$（环境转移动力学不依赖 $\theta$）
- 只有 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 留了下来

因此，轨迹对数概率的梯度化简为：
$$
\nabla_\theta \log p_\theta (\tau) = \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t)
$$ {#eq:trajectory_log_grad}

**推导到这一步，是实现上的一个关键节点。**
到这里我们已经看得足够远：轨迹分布的梯度化简成了语言模型策略概率（也就是我们正在训练的模型给出的 token 概率）的梯度之和。
实践中，这产生了策略梯度方程的一种常见形式。
它们最终在损失里看起来像是对数概率求和，然后我们通过自动微分计算梯度。
你会反复看到的一段简短代码大致如下：

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

本章通篇你都会看到它。现在，回到形式化的策略梯度数学。

把它代回 @eq:policy_gradient_expectation，我们得到：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^\infty R(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$ {#eq:policy_gradient_returns}

人们经常使用策略梯度的一个更一般的表述：
$$
g = \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^\infty \Psi_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$ {#eq:general_gradient}

其中 $\Psi_t$ 可以是下面这些（奖励通常也可以用 $\gamma$ 折扣）——这套分类沿用了 Schulman 等 2015 年的做法 [@schulman2015high]：

1. $R(\tau) = \sum_{t=0}^{\infty} r_t$：整条轨迹的总奖励。
2. $\sum_{t'=t}^{\infty} r_{t'}$：动作 $a_t$ 之后的奖励，也叫从时刻 $t$ 起的回报 $G_t$。
3. $\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$：上一式带 baseline 的版本。
4. $Q^{\pi}(s_t, a_t)$：状态-动作值函数。
5. $A^{\pi}(s_t, a_t)$：优势函数；如果能被准确计算，它给出理论上最低的方差。
6. $r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$：时序差分（TD）残差。

*baseline* 是一个用来降低策略更新方差的量（下面会详述）。

对语言模型来说，其中一些概念没那么说得通。
例如，对确定性策略 $\pi$，状态价值是 $V^{\pi}(s_t) = Q^{\pi}(s_t, \pi(s_t))$（对最优价值函数则有 $V^*(s_t)=\max_{a_t} Q^*(s_t,a_t)$）。对随机策略，对应的恒等式是 $V^{\pi}(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot\mid s_t)}\!\left[Q^{\pi}(s_t,a_t)\right]$。
贝尔曼方程把 Q 与 V 联系起来：一般来说 $Q^\pi(s_t,a_t) = \mathbb{E}\!\left[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t\right]$；但对状态转移确定性的语言模型，这化简为 $Q(s_t,a_t) = r_t + \gamma V(s_{t+1})$。
优势函数衡量动作 $a_t$ 比平均水平好多少：

$$A(s_t,a_t) = Q(s_t,a_t) - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$ {#eq:advantage_trick}

最后这个形式正是时序差分（TD）残差（上面第 6 项）——RL 中的一个基本量，衡量价值函数的预测与实际发生之间的差距，驱动价值函数向更准确的估计更新。实践中，我们用学到的价值函数 $\hat{V}$ 通过这个 TD 误差来估计优势。

### 朴素策略梯度

朴素策略梯度的实现，是对上述 $J(\theta)$ 关于策略参数求导来优化。
一个简单的版本——关于时刻 $t$ 的回报——是：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T G_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$ {#eq:vanilla_policy_gradient}

朴素策略梯度算法的一个常见问题是梯度更新方差很大，这可以用多种方式缓解。
高方差源于：梯度更新是用回报 $G$ 的估计算出来的，而这个估计往往只来自环境中一小批 rollout，天然容易受噪声影响（例如语言模型在温度 $>0$ 下生成时固有的随机性）。
在奖励稀疏的领域，回报估计之间的方差更高，因为更多样本是 0 或 1，而不是紧密聚集在一起。
为缓解这一点，人们用各种技术来归一化价值估计，称为 *baseline*。
Baseline 以多种方式做到这件事，本质上是按“状态相对于下游动作的价值”来做归一化（例如 Advantage 就是 Q 值与 V 值之差）。
最简单的 baseline 是奖励的批均值或滑动平均。
即便是这些**与动作无关**的 baseline，也能在不改变期望梯度的前提下降低方差，因为对任意只依赖状态的 $b(s)$ 都有 $\mathbb{E}_{a \sim \pi(a|s)}\!\left[b(s) \nabla_\theta \log \pi_\theta(a|s)\right] = 0$，从而显著改善学习信号。

本章讨论的许多策略梯度算法，都建立在策略梯度的优势形式上：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T A^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$ {#eq:advantage_policy_gradient}


### REINFORCE

REINFORCE 这个名字很可能是先有词、后凑出的缩写（backronym），但它所代表的算法组件对现代强化学习算法相当重要。
它定义于奠基性论文 *Simple statistical gradient-following algorithms for connectionist reinforcement learning* [@williams1992simple]：

> 这个名字是 “REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility” 的缩写。

这三个组件对应的就是如何完成*奖励增量*，也就是策略梯度步。
它的更新规则包含三部分：

1. 非负因子：学习率（步长），必须是正数，例如下面的 $\alpha$。
2. Offset Reinforcement（偏置强化）：一个 baseline $b$，或对奖励的其他归一化因子，用来提升稳定性。
3. Characteristic Eligibility（特征资格）：把标量奖励信号归因到产生该动作的参数上。Williams 用 $e$ 表示这个资格项（不是指数函数）。在现代策略梯度记号中，它对应 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$。

于是形式看起来很熟悉：

$$ \Delta_\theta = \alpha(r - b)e $$ {#eq:REINFORCE_BASIC}

换成更现代的记号与广义回报 $G$，REINFORCE 算子写作：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim p_{\theta}}\!\left[
    \sum_{t=0}^{T}
    (G_t - b(s_t))\,\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
\right],
$$ {#eq:REINFORCE_with_baseline}

这里 $G_t - b(s_t)$ 就是策略在当前状态下的*优势*，所以我们可以把策略梯度改写成一种后面会继续使用的、带优势 $A$ 的形式：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim p_{\theta}}\!\left[
    \sum_{t=0}^{T}
    A_t\,\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)
\right],
$$ {#eq:REINFORCE_with_advantage}

REINFORCE 是朴素策略梯度的一个具体实现，它使用梯度的蒙特卡洛估计量。

![面向语言模型的基础 REINFORCE 架构。经过整形的奖励把奖励模型分数与来自参考模型的 KL 惩罚结合在一起。本章后续都在这个结构上搭建。](images/reinforce_tikz.png){#fig:reinforce-arch data-dark-src=“images/reinforce_tikz-dark.png”}

### REINFORCE Leave One Out（RLOO）

REINFORCE Leave One Out 与标准 REINFORCE 的核心实现差异在于：它用批次中*其他*样本的平均奖励来算 baseline，而不是对批次里所有奖励取平均 [@huang2024putting]、[@ahmadian2024back]、[@kool2019buy]。
通过把当前样本自己的奖励排除在它自己的 baseline 之外，RLOO 的 baseline 与所评估的动作相互独立，从而让梯度估计量保持**严格无偏**。

关键在于，这种方法只在“每个状态（提示）生成多条轨迹（补全）”时才成立——而这在用 RL 微调语言模型的多个领域中都是常见做法。

具体来说，对 REINFORCE Leave-One-Out（RLOO）的 baseline，给定对某个提示 $s$ 采样的 $K$ 条轨迹（在提示条件下采取的动作）$a_1, \dots, a_K$，我们把 baseline 显式定义为如下**按提示**的量：

$$
b(s, a_k) = \frac{1}{K-1}\sum_{i=1, i\neq k}^{K} R(s, a_i),
$$ {#eq:RLOO_baseline}

于是优势为：

$$
A(s, a_k) = R(s, a_k) - b(s, a_k).
$$ {#eq:RLOO_advantage}

等价地，这可以表达为：

$$
A(s, a_k) = \frac{K}{K - 1}\left(R(s, a_k) - \frac{1}{K}\sum_{i=1}^{K} R(s, a_i)\right).
$$ {#eq:RLOO_advantage_alt}

这是一个简单、低方差的**按提示**优势估计，与组相对策略优化（GRPO）中使用的组相对优势密切相关（GRPO 稍后讲，在近端策略优化 PPO 之后）。
实践中，GRPO 式训练的主要差别在于它如何施加 KL 正则（作为显式的损失项，还是折进奖励里），以及是否使用 PPO 式的比率裁剪。
具体地说，经典的 GRPO 实现把 KL 惩罚加在损失层面，而 RLOO 或传统策略梯度的推导则把 KL 惩罚加在奖励本身上。
随着从 RLHF 转向推理与可验证奖励的强化学习（RLVR），KL 惩罚的普遍性整体下降，许多对 RLHF 代码的推理化改造干脆把它整个关掉。
不过，RLOO 的优势仍可与 PPO 的裁剪结合使用——这也说明这些算法彼此有多相似。

RLOO 以及其他不使用价值网络的算法——价值网络是额外的一份模型副本（一个 critic），逐 token 预测标量价值 $V(s_t)$——在计算损失时把同一个序列级优势（或奖励）分配给每个 token。
而使用学到的价值网络的算法（如 PPO）为每个 token 单独分配不同的值，从 EOS token 处获得的最终奖励往回折扣。
配上 KL 距离惩罚时，RLOO 把逐 token 的 KL 在整个补全上聚合，再把这个标量折进序列奖励，于是得到的优势被广播到所有 token。
PPO 则在计算 $A_t$ 之前，从逐 token 奖励中减去逐 token 的 KL，从而实现 token 级的信用分配。
GRPO 通常保留序列级优势，但在损失中额外增加一个逐 token 的项，而不是从奖励中减去它。
这些细节与权衡将在本章后文讨论。

![REINFORCE Leave-One-Out（RLOO）架构。每个提示生成多个补全，从而无需学习价值函数就能得到用于优势估计的留一 baseline。](images/rloo_tikz.png){#fig:rloo-arch data-dark-src=“images/rloo_tikz-dark.png”}

<!-- A nice formulation of LM RL loss functions is found here https://arxiv.org/pdf/2502.01600 -->

### 近端策略优化（PPO）

近端策略优化（PPO）[@schulman2017proximal] 是深度 RL 众多成功背后的奠基性算法之一（例如掌握 Dota 2 的 OpenAI Five [@berner2019dota]，以及大量研究）。
PPO 最大化、关于优势与策略概率的目标如下：

$$J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).$$ {#eq:PPO_EQN}

这里 $\pi_\theta(a|s)$ 是正在被优化的当前策略，$\pi_{\theta_{\text{old}}}(a|s)$ 是用于收集训练数据的策略（即上一轮迭代的策略）。
这两个策略的比值来自*重要性采样*——它让我们能够复用旧策略下收集的数据来估计新策略的梯度。

回顾策略梯度的优势形式（@eq:advantage_policy_gradient），我们有：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta} \left[ \sum_{t=0}^T A^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right].$$ {#eq:advantage_policy_gradient_recall}

这个期望是在由 $\pi_\theta$ 诱导的轨迹分布上取的；但实践中我们想在一个由固定策略 $\pi_{\theta_{\text{old}}}$ 收集的批次上做多个梯度步。
为修正这个分布不匹配，我们乘上重要性权重 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$，它把样本重新加权，以反映它们在当前策略下相对于数据收集策略变得多可能或少可能。
如果不加约束，优化这个重要性加权目标会在比值远离 1 时导致破坏性的巨大策略更新。
PPO 的应对方式是把比值裁剪到区间 $[1-\varepsilon, 1+\varepsilon]$，确保策略在单次更新中不会变化过猛。

注意，当我们进入 PPO 及其同类算法时，我们往往处理*目标函数*而不是显式梯度。
原因是：一旦加入 $\min$ 与裁剪操作，PPO 目标就**没有**一个易解读的解析梯度了（视写法不同，梯度大约有 4 项，对应 @fig:ppo-obj 中的各个区域）；写出目标函数是传达这些算法更清晰的方式。

为完整起见，PPO 通常写成在时间步上的*期望*裁剪代理目标：

$$
J(\theta)
=
\mathbb{E}_{t}\left[
\min\left(\rho_t(\theta)A_t,\ \text{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)A_t\right)
\right],
\qquad
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.
$$ {#eq:PPO_EQN_EXPECTED}

这个目标通常只需加一个负号就转成损失函数，从而让优化器去把它压得越低越好。

对语言模型而言，目标（或损失）是**逐 token** 计算的；这可以直观地建立在“整段自回归预测序列的概率如何计算”之上——即概率的连乘。
由此，常见实现是用*对数概率*，因为在现代语言建模框架里这样做计算更简单。
实践中，人们计算 token 对数概率之差，再取指数，就还原出策略比值 $\rho_t$。

$$ J(\theta) = \frac{1}{|a|} \sum_{t=0}^{|a|} \min\left(\frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}A_{t}, \text{clip} \left( \frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}, 1-\varepsilon, 1+\varepsilon \right) A_{t} \right).  $$  {#eq:PPO_EQN_EXPANDED}

这是 PPO 的逐 token 版本，也同样适用于其他策略梯度方法；本章后面的实现部分会进一步展开。
这里的“按动作 token 数取平均”那一项 $\frac{1}{|a|}$ 来自常见的实现习惯，并不在损失的正式推导中（见 [@liu2025understanding]）。

![PPO 框架。一个学到的价值函数支撑广义优势估计（GAE）以得到逐 token 优势，再配合裁剪代理目标使用。](images/ppo_tikz.png){#fig:ppo-arch data-dark-src=“images/ppo_tikz-dark.png”}

下面我们解释这个损失函数在不同优势与策略比值下会触发哪些情形。
在实现层面，PPO 的内部计算涉及两个主要项：1）带学到优势的标准策略梯度；2）基于最大步长的裁剪策略梯度。

为理解各种情形如何出现，我们可以把策略比值定义为：

$$\rho(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$ {#eq:PPO_POL_RATIO}

策略比值是 PPO 及相关算法的核心。
它来自对策略求梯度的过程，并以非常直观的方式控制参数更新。
对任意一批数据，该批次的第一个梯度步中策略比值起始为 1，因为此时 $\pi_{\theta}$ 与 $\pi_{\theta_{\text{old}}}$ 相同。随后在下一个梯度步中，如果该步提高了某些具有正优势的 token 的可能性，比值就会大于 1；反之则小于 1。常见做法是在更新 $\pi_{\theta_{\text{old}}}$ 之前，对每批数据用策略梯度算法做 1–4 个梯度步。

#### 理解 PPO 的目标函数

总体而言，PPO 的目标可以由图 @fig:ppo-obj 中“目标对策略比值”的两条曲线来可视化。
PPO 的目标通过改变被采样动作的概率而被最大化。
在数值上，这个目标用巧妙的 $\min$ 操作同时处理正负优势两种情形，使得更新最多只会把策略比值从 1 推开 $\varepsilon$ 的距离。

**在信任域内部，PPO 与其他策略梯度算法的作用相同。**
这是有意设计的！信任域是一个用来给 PPO 及其同类算法的最大步长封顶、以保证更新稳定性的概念。PPO 算法的核心——裁剪与 min/max 函数——定义的正是这个区域。在它之外，目标变为平坦。

“信任域”这个想法来自数值优化文献 [@nocedal2006numerical]，但在深度 RL 中是由信任域策略优化（TRPO）算法普及开来的，后者被公认为 PPO 的前身 [@schulman2015trust]。
信任域就是完整策略梯度步被施加的区域——因为这里的更新不会被 PPO 目标的 max/min 操作“裁剪”掉。

![PPO 目标 $J(\theta)$ 随策略比值 $\rho(\theta)$ 变化的可视化，分别对应正优势与负优势。每个子图中，三个比值区间都标注了未裁剪项、裁剪项、得到的目标以及梯度。](images/ppo-clip-viz.png){#fig:ppo-obj}

策略比值与优势的组合会落在几种不同构型中；@fig:ppo-obj 按优势 $A_t$ 的符号、以及策略比值 $\rho(\theta)$ 落入三个区间中的哪一个来枚举。
每个区间的结果由两个事实决定：优势的符号决定我们想让该动作更可能还是更不可能，而 $\min$ 操作则在未裁剪项 $\rho(\theta) A_t$ 与其裁剪版本之间做选择。

裁剪**只在两个区域**把梯度归零——即策略**已经**把被采样动作朝期望方向推过了信任域边缘的区域：

- **正优势且 $\rho(\theta) > 1+\varepsilon$**：在 $\pi_\theta$ 下该动作已经比在 $\pi_{\theta_{\text{old}}}$ 下明显更可能。目标在 $(1+\varepsilon)A_t$ 处饱和，梯度为零，不做更新——我们避免对一个已经过度表达的动作继续过度强化。
- **负优势且 $\rho(\theta) < 1-\varepsilon$**：在 $\pi_\theta$ 下该动作已经明显更不可能。目标在 $(1-\varepsilon)A_t$ 处饱和，梯度同样为零，不做更新——我们避免对一个已经被抑制的动作继续过度压制。

其余各处，未裁剪项 $\rho(\theta) A_t$ 生效，PPO 采取标准的策略梯度步：当 $A_t > 0$ 时提高该动作的概率，当 $A_t < 0$ 时降低它。我们可以按“每个区域对更新后的策略 $\pi_\theta$ 提出了什么要求”来读 @fig:ppo-obj：

- 正优势下的斜线、未裁剪区域（绿色）**提高**被采样动作的概率；
- 负优势下的斜线、未裁剪区域（红色）**降低**它；
- 平坦的裁剪区域（灰色）让策略**保持不变**，因为其梯度为零。

把同样的区域逐项写出来：

##### 正优势（$A_t > 0$）

这意味着根据价值函数，所采取的动作是有益的，我们想在将来提高采取该动作的可能性。下面看策略比值 $\rho(\theta)$ 的各种情形：

1. $\rho(\theta) < 1 - \varepsilon$：

    - **解释**：新策略下该动作比旧策略下更不可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 - \varepsilon) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常的策略梯度更新——提高该动作的可能性

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$：

    - **解释**：新策略下该动作与旧策略下几乎同样可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$\rho(\theta) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常的策略梯度更新——提高该动作的可能性

3. $1 + \varepsilon < \rho(\theta)$：

    - **解释**：新策略下该动作比旧策略下更可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 + \varepsilon) A_t$
    - **目标**：$(1 + \varepsilon) A_t$
    - **梯度**：$\nabla_\theta (1 + \varepsilon) A_t = 0$
    - **发生什么**：不更新——该动作在新策略下已经更可能

总结一下：当优势为正（$A_t>0$）时，我们想提升该动作的概率。因此：

- 只有当 $\pi_{\text{new}}(a) \leq (1+\varepsilon) \pi_{\text{old}}(a)$ 时才做梯度步。直观地说，由于优势为正，我们想提升该动作的概率，但不要提得太猛，以至于把它变得明显更可能。
- 关键在于，当 $\pi_{\text{new}}(a) > (1+\varepsilon) \pi_{\text{old}}(a)$ 时，我们不做任何更新，裁剪目标的梯度为 $0$。直观地说，该动作在新策略下已经过度表达，我们不想继续过度强化它。

##### 负优势（$A_t < 0$）

这意味着根据价值函数，所采取的动作是有害的，我们想在将来降低采取该动作的可能性。下面看策略比值 $\rho(\theta)$ 的各种情形：

1. $\rho(\theta) < 1 - \varepsilon$：

    - **解释**：新策略下该动作比旧策略下更不可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 - \varepsilon) A_t$
    - **目标**：$(1 - \varepsilon) A_t$
    - **梯度**：$\nabla_\theta (1 - \varepsilon) A_t = 0$
    - **发生什么**：不更新——该动作在新策略下已经更不可能

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$：

    - **解释**：新策略下该动作与旧策略下几乎同样可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$\rho(\theta) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常的策略梯度更新——降低该动作的可能性

3. $1 + \varepsilon < \rho(\theta)$：

    - **解释**：新策略下该动作比旧策略下更可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 + \varepsilon) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常的策略梯度更新——降低该动作的可能性

总结一下：当优势为负（$A_t < 0$）时，我们想降低该动作的概率。因此：

- 只有当 $\pi_{\text{new}}(a) \geq (1-\varepsilon) \pi_{\text{old}}(a)$ 时才做梯度步。直观地说，由于优势为负，我们想降低该动作的概率，并且降低的程度与优势成比例。
- 关键在于，当 $\pi_{\text{new}}(a) < (1-\varepsilon) \pi_{\text{old}}(a)$ 时，我们不做任何更新，裁剪目标的梯度为 $0$。直观地说，该动作在新策略下已经更不可能，我们不想继续过度压制它。

**务必记住：信任域内部的 PPO 与标准形式的策略梯度大致相同。**


#### 价值函数与 PPO

PPO 中的价值函数是模型的额外一份副本，用来预测每个 token 的价值。
在传统 RL 中，一个 token（或状态）的价值预测的是从该时刻起的未来回报，通常带折扣。
PPO 中的这个价值被用作学到的 baseline，代表了对 REINFORCE 所用简单蒙特卡洛版本的演进（后者不需要学到的价值网络）。
这也凸显出：PPO 在优化形式、baseline 等多个方面都是 REINFORCE 与朴素策略梯度的演进。
实践中，对 PPO 及语言模型所用的其他算法而言，这个价值预测的是每个 token 在**扣除 KL 惩罚之后**的回报（如前所述，逐 token 损失传统上包含来自奖励的 KL）。

学习价值函数有几种不同的方法（或目标）。
广义优势估计（GAE）被认为是现代系统中的最先进且经典的做法，但它更复杂——要在多个步上计算价值预测误差，见本章后面关于 GAE 的一节。
价值函数也可以用“更新策略所用 rollout”的蒙特卡洛估计来学习。
PPO 有两个损失——一个用来学习价值函数，另一个用该价值函数来更新策略。

![价值函数的训练使用同策略 rollout 计算目标。模型在每个 token 处预测 $V_t$，并通过 MSE 针对目标回报 $\hat{V}_t$ 训练。随后优势 $A_t = \hat{V}_t - V_t$ 为策略梯度更新加权。](images/value_fn_training.png){#fig:value_fn_training data-dark-src=“images/value_fn_training-dark.png”}

一个价值网络损失的简单示例实现如下。

```python
# Basic PPO critic targets & loss (no GAE)
#
# B: Batch Size
# L: Completion Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards; EOS row includes outcome
#   done_mask: (B, L) 1.0 at terminal token (EOS or truncation if penalized), else 0.0
#   completion_mask: (B, L) 1.0 on response tokens to supervise (ignore the prompt)
#   values: (B, L) current critic predictions V_theta(s_t)
#       because a value network is a running update
#   old_values: (B, L) critic predictions at rollout time V_{theta_old}(s_t)
#   gamma: discount factor, float (often 1.0 for LM RLHF)
#   epsilon_v: float value clip range (e.g., 0.2), similar to PPO Loss Update itself, optional
#
# Returns:
#   value_loss: scalar; advantages: (B, L) detached (for policy loss)

B, L = rewards.shape

# 1) Monte Carlo returns per token (reset at terminals)
# Apply discounting, if enabled
returns = torch.zeros_like(rewards)
running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(L)):
    running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
    returns[:, t] = running

targets = returns  # y_t = G_t (post-KL)

# 2) PPO-style value clipping (optional)
v_pred = values
v_old  = old_values
v_clip = torch.clamp(v_pred, v_old - epsilon_v, v_old + epsilon_v)

vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss_tok  = torch.max(vf_unclipped, vf_clipped)

# 3) Mask to response tokens and aggregate
denom = completion_mask.sum(dim=1).clamp_min(1)
value_loss = ((vf_loss_tok * completion_mask).sum(dim=1) / denom).mean()

# 4) Advantages for policy loss (no GAE): A_t = G_t - V(s_t)
advantages = (targets - v_pred).detach()

# The value loss is applied later, often with the PG loss, e.g.
# total_loss = policy_loss + vf_loss_coef * value_loss
```

### 组相对策略优化（GRPO）

组相对策略优化（GRPO）由 DeepSeekMath 提出 [@shao2024deepseekmath]，并用于 DeepSeek 的其他工作，例如 DeepSeek-V3 [@deepseekai2025deepseekv3technicalreport] 与 DeepSeek-R1 [@guo2025deepseek]。
GRPO 可以看作一个受 PPO 启发的算法：代理损失非常相似，但它不需要用原策略语言模型的另一份副本（或另一个检查点作初始化）去学习价值函数。
这带来两个公认的好处：

1. 避开了“从 LM 主干学价值函数”这一难题——该领域的研究尚未确立最佳实践。
2. 省显存：不需要在内存里保留额外一套模型权重（从需要当前策略、参考策略和价值函数三份，变成只需前两份）。

GRPO 做到这一点的方式是简化价值估计：通过估计优势或 baseline，它给回合中每个 token 赋同一个值（即对某个提示的补全，每个 token 得到相同的值，而不是标准价值函数里的折扣奖励）。
这个估计来自从同一初始状态/提示（$s$）收集多个补全（$a_i$）和奖励（$r_i$），也就是蒙特卡洛估计。

形式化地说，GRPO 目标与上面的 PPO 目标非常相似。
对 GRPO，目标（或损失）是在给定提示 $s$ 的一组补全 $\{a_1, a_2, ..., a_G\}$ 上累加的。
GRPO 目标如下：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right).$$ {#eq:GRPO}

注意，相对于 PPO，GRPO 的标准实现把 KL 距离放在损失里。
与上面类似，我们可以把它展开成逐 token 的计算：

$$\begin{aligned}
J(\theta) = \frac{1}{G}\sum_{i=1}^G  \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \Bigg( &\min\!\left(\frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}A_{i,t},\; \text{clip} \left( \frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}, 1-\varepsilon, 1+\varepsilon \right) A_{i,t} \right) \\
&- \beta \mathcal{D}_{\text{KL}}\!\left(\pi_\theta(\cdot|s_{i})\|\pi_{\text{ref}}(\cdot|s_{i})\right) \Bigg)
\end{aligned}$$ {#eq:GRPO_token}


补全下标 $i$ 的优势计算为：

$$A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}.$$ {#eq:GRPO_ADV}

![GRPO 架构。优势相对于组内均值与标准差做归一化。KL 惩罚直接施加在损失上，而不是用来整形奖励。](images/grpo_tikz.png){#fig:grpo-arch data-dark-src=“images/grpo_tikz-dark.png”}

直观上，GRPO 的更新是在一个批次内比较对同一个问题的多个答案。
模型学会变得更像那些被标为正确的答案、更不像其他答案。
这是计算优势的一种非常简单的方式——优势衡量的是某个具体动作在给定状态下比平均水平好多少。
相对于 PPO、REINFORCE，以及广义上“用奖励模型打分（相对输出奖励）的 RLHF”，GRPO 通常对每个提示采样**多得多的样本**，因为它的优势完全取决于一个补全相对于同一提示下其他补全的价值。
这里，当前策略针对给定提示生成多个回复，而组式的 GRPO 优势估计因此获得了有价值的上下文。
PPO 与朴素策略梯度算法本来是为了准确估计每个补全的奖励而设计的（事实上，在某些情况下更多补全对改善价值估计帮助不大）。
GRPO 及其变体特别契合现代语言模型工具——在那里，对给定提示取多个补全非常自然（尤其是和“机器人任务中从某个固定环境状态出发的多个动作”相比）。

GRPO 的优势计算在偏置上有取舍。
按标准差做归一化，会**奖励**一个批次中答案正确性变化小的问题。
对于答案几乎全对或全错的问题，标准差更小，优势更大。
Liu 等 2025 [@liu2025understanding] 基于这一偏置提出去掉标准差项，但代价是会**降低**“大部分错、少数对”这类问题的权重——而那可能恰恰是对模型有价值的学习信号。
这些高方差提示可能正是最难的案例：只有少数采样补全找到了正确答案，从而提供了很强的训练信号。

@eq:GRPO_ADV 是 GRPO 在结果监督下（标准奖励模型，或单一可验证奖励）的实现；在过程监督下则需要不同的实现。
那种情况下，GRPO 把优势计算为其后各推理步骤归一化奖励之和。

最后，GRPO 的优势估计也可以不配 PPO 裁剪，用在更朴素的策略梯度版本上（例如 REINFORCE），但这不是其经典形式。
作为这些算法彼此交织的一个例子，我们可以说明：GRPO 的一个变体 Dr. GRPO（GRPO Done Right）[@liu2025understanding] 的优势估计，与 RLOO 估计（用其他样本的平均奖励作 baseline）在相差一个常数缩放因子意义下等价（由于实现中通常会对优势做归一化，这个常数一般无关紧要）。
Dr. GRPO 从 @eq:GRPO_ADV 中去掉了标准差归一化项——注意这同时把优势*放大*了，效果等价于对“答案分数有方差”的样本提高了 GRPO 的学习率。
这解决了一个偏向“奖励方差小的问题”（即答案几乎全对或全错）的偏置，但如果“只有一条样本答对的问题”本身很重要，这样做也有潜在代价。
对规模为 $G$ 的组内补全 $i$，Dr. GRPO 的优势定义为：

$$ \tilde{A}_i = r_i - \text{mean}({r_1, r_2, \cdots, r_G}) = r_i - \frac{1}{G}\sum_{j=1}^G r_j $$ {#eq:DrGRPO_ADV}

用同样的记号，我们可以回忆 RLOO 的优势估计为：

$$ A_i^\text{RLOO} = r_i - \frac{1}{G-1}\sum_{j=1, i\neq j}^G r_j $$ {#eq:RLOO_ADV_AGAIN}

于是，如果把 Dr. GRPO 的优势定义乘上 $\frac{G}{G-1}$，就能看出一种缩放等价：

$$
\begin{aligned}
\frac{G}{G-1} \tilde{A}_i &= \frac{G}{G-1} \left( r_i - \frac{1}{G}\sum_{j=1}^G r_j \right) \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1}^G r_j \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j - \frac{1}{G-1} r_i \\
&= r_i \left( \frac{G}{G-1} - \frac{1}{G-1} \right) - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= A_i^{\text{RLOO}}
\end{aligned}
$$ {#eq:RLOO_GRPO_EQUIV}

### 组序列策略优化（GSPO）

当在“由先前策略收集的一批数据”上做多个梯度步时，需要用重要性采样来修正数据收集策略与当前被优化策略之间的分布不匹配。
标准的重要性采样恒等式让我们能用来自一个分布的样本估计另一个分布下的期望：

$$
\mathbb{E}_{p}[f(x)] = \mathbb{E}_{q}\left[f(x) \frac{p(x)}{q(x)}\right],
$$ {#eq:IS_identity}

其中 $p$ 是目标分布，$q$ 是采样分布，$\frac{p(x)}{q(x)}$ 是重要性权重。
在策略梯度方法中，$p = \pi_\theta$ 是我们要优化的当前策略，$q = \pi_{\theta_{\text{old}}}$ 是生成训练数据的策略。
这让我们能对在 $\pi_{\theta_{\text{old}}}$ 下收集的样本重新加权，以估计 $\pi_\theta$ 的梯度，从而支持每批 rollout 做多个梯度步。

这种分布不匹配出现在两种常见场景中：(1) 对单个批次做多个梯度步，此时 $\pi_\theta$ 在每次更新后都会偏离 $\pi_{\theta_{\text{old}}}$；(2) 在异步训练系统中，推理后端（如 vLLM）与训练后端（如 FSDP）可能因同步延迟而持有不同的模型权重（见本章后面的「异步性」一节——这一做法尤其在 RLVR 兴起后变得普遍，但在 RLHF 设定中也有使用）。

PPO 与 GRPO 在 **token 层面**施加重要性采样，并通过裁剪*代理目标*来稳定学习。
然而这种做法有一个微妙的失效模式：当某个 token 的重要性比值超出裁剪范围 $[1-\varepsilon, 1+\varepsilon]$ 时，该 token 就得到零梯度。
对于那些罕见但重要的 token——例如模型起初赋予低概率的关键推理步骤——这种“token 丢弃”会阻碍模型学会更可靠地产生它们。

组序列策略优化（GSPO）[@zheng2025gspo] 扩展了 GRPO：它在**序列层面**而非 token 层面计算重要性比值。
这个算法的实践动机——以及它的同类 CISPO（后者改变了策略梯度算法中重要性采样的计算方式，稍后讨论）——在于逐 token 的重要性比值往往数值不稳定。
其概念动机则是：当奖励在序列层面被赋予时（多数 RLHF 与 RLVR 设定都是如此），重要性采样修正也应当与那个粒度匹配。

对长序列和/或大型稀疏模型（例如现代专家混合 MoE 模型），逐 token 比值的行为可能很不稳定：单个比值很大的 token 就可能主导策略更新，或者一条回复里许多 token 各自被独立裁剪，把学习信号在同一回复内打碎。
GSPO 通过为每条回复计算单个重要性权重来解决这个问题。

回忆一下，一条完整回复的概率按自回归方式分解：

$$
\pi_\theta(a \mid s) = \prod_{t=1}^{|a|} \pi_\theta(a_t \mid s, a_{<t}).
$$ {#eq:response_factorization}

为简化起见，我们常把条件策略 $\pi_\theta(a_t \mid s, a_{<t})$ 简写为 $\pi_\theta(a_t \mid s)$，其中隐含了补全中之前的动作（token）。
GSPO 用几何平均定义了一个按长度归一化的序列级重要性比值（以避免长序列带来的数值问题）：

$$
\rho_i(\theta) = \left( \frac{\pi_\theta(a_i \mid s)}{\pi_{\theta_{\text{old}}}(a_i \mid s)} \right)^{\frac{1}{|a_i|}} = \exp\left( \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s, a_{i,<t})} \right).
$$ {#eq:GSPO_ratio}

GSPO 的目标与 GRPO 类似，只是使用了这个序列级比值：

$$
J_{\text{GSPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i(\theta) A_i,\, \text{clip}(\rho_i(\theta), 1-\varepsilon, 1+\varepsilon) A_i \right) \right].
$$ {#eq:GSPO_objective}

由于比值经过长度归一化，裁剪范围 $\varepsilon$ 作用在“逐 token 平均”的尺度上，使有效约束在不同长度的回复之间可比。
实现时，序列级权重 $\rho_i$ 被统一施加到回复 $a_i$ 中的所有 token 上——这简化了梯度计算，同时保持了序列级的重要性采样修正。

优势计算与 GRPO 相同（@eq:GRPO_ADV），使用组相对均值与标准差归一化，也可以像 GRPO 的其他衍生研究那样加以修改。
GSPO 可以概括为“带序列级重要性比值的 GRPO”——重要性采样修正的粒度与奖励的粒度相匹配。

### 裁剪重要性采样策略优化（CISPO）

裁剪重要性采样策略优化（CISPO）[@minimax2025minimax_m1] 走了另一条路：它不裁剪代理目标，而是**直接裁剪重要性权重本身**，同时为所有 token 保留梯度。
该目标对被裁剪的重要性权重使用 stop-gradient，从而回到 REINFORCE 式的表述，而不是 PPO 式的双边裁剪：

$$
J_{\text{CISPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^K \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{\sum_{i=1}^K |a_i|} \sum_{i=1}^K \sum_{t=1}^{|a_i|} \text{sg}\left( \hat{\rho}_{i,t}(\theta) \right) A_{i,t} \log \pi_\theta(a_{i,t} \mid s, a_{i,<t}) \right],
$$ {#eq:CISPO_objective}

其中 $\text{sg}(\cdot)$ 表示 stop-gradient（该权重被使用但不参与求导），被裁剪的重要性比值为：

$$
\hat{\rho}_{i,t}(\theta) = \text{clip}\left( \rho_{i,t}(\theta),\, 1 - \varepsilon_{\text{low}},\, 1 + \varepsilon_{\text{high}} \right), \quad \rho_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s, a_{i,<t})}.
$$ {#eq:CISPO_ratio}

它与 PPO/GRPO 的关键差别微妙但重要：**裁剪权重（而非目标）**意味着每个 token 仍然获得一个与其优势成正比的梯度信号——权重只是限制了这个信号被重要性比值放大或压制的幅度。
这是一个偏差-方差权衡：裁剪权重引入偏差，但控制了方差，而且关键在于避免了把 token 梯度完全丢弃。

**CISPO 与 GSPO 都是由那些在把 RL 应用到大规模 MoE 模型上推进极限的机构提出的**，而 MoE 模型以数值问题著称。
相关论文指出，逐 token 的重要性采样比值不稳定，会给梯度带来可观的方差、拖累学习。
这让这些算法在大规模模型上格外有影响，但在更小规模的学术实验中研究得更少、收益也更不明显。

CISPO 还允许非对称的裁剪边界（$\varepsilon_{\text{low}} \neq \varepsilon_{\text{high}}$），类似于本章后面会讨论的 DAPO 的“clip-higher”改动——它可以为模型想要提高权重的 token 允许更大的更新，从而鼓励探索。
相关工作包括 Tapered Off-Policy REINFORCE（TOPR）[@leroux2025topr]，它也像 CISPO 那样直接裁剪重要性采样权重，而不是像 PPO/GRPO 那样在目标函数内部裁剪；但它工作在序列层面（像 GSPO），并依据奖励符号使用非对称裁剪——对正奖励不做重要性采样修正，对负奖励把比值裁剪到 $[0, 1]$——从而实现稳定的异策略学习。


### 算法对比

本章每个算法共享同一个核心梯度形状（@eq:policy_gradient_intuition），但在如何估计优势、如何控制优化上不同：

- **REINFORCE**：最简单的策略梯度实现，用奖励的蒙特卡洛估计与基于状态的 baseline 来降低方差。
- **RLOO**：REINFORCE 加每个提示多样本，每个样本的 baseline 取其他样本奖励的平均（留一法），以降低梯度方差。
- **PPO**：增加一个学到的价值函数和一个裁剪策略比值，以获得更准确、更稳定的梯度更新。
- **GRPO**：PPO 的简化变体——对每个提示的多个补全分组，在组内归一化奖励来计算优势，从而不需要价值函数。
- **CISPO**：REINFORCE 式算法，对重要性采样**权重**（而不是像 PPO/GRPO 那样对目标）做带 stop-gradient 的裁剪以求稳定，因此每个 token 都能收到梯度信号。
- **GSPO**：类似 GRPO，但把策略比值按补全长度归一化，从而防止长度偏置。
- **DPO**：不是 RL 算法，而是一种绕过独立奖励模型、直接从偏好对优化、以解决同一偏好优化问题的方法（见第 8 章）。

上述所有策略梯度算法在推导上都是**同策略**的，尽管实践中大多数会被略微用于异策略场景。第 8 章的 DPO 及其他直接对齐算法默认就是**异策略**的。
它们都可以搭配学到的奖励模型或可验证奖励。
只有 PPO 需要一个学到的价值函数。
REINFORCE 与 RLOO 没有重要性采样比值——其余算法各自引入一个，以便在一批 rollout 上做多个梯度步，其差别在于粒度和裁剪策略，汇总如下表。

| 方法 | 重要性采样粒度 | 裁剪方式 | 优势 |
| :----- | :-----------: | :------------------: | :-------------------: |
| **REINFORCE** | 无 | 无 | 蒙特卡洛 baseline |
| **RLOO** | 无 | 无 | 留一法 |
| **PPO** | token | 目标（双边） | 学到的价值函数 |
| **GRPO** | token | 目标（双边） | 组相对 |
| **GSPO** | 序列 | 目标（双边） | 组相对 |
| **CISPO** | token | 权重（stop-grad） | 组相对 |
Table: 策略梯度算法对比。 {#tbl:pg_compare}

各方法的核心损失 $\mathcal{L}(\theta)$ 为：

$$\begin{aligned}
\textbf{REINFORCE:}\quad & -\frac{1}{T}\sum_{t=1}^{T}\log \pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big) \\[6pt]
\textbf{RLOO:}\quad & -\frac{1}{K}\sum_{i=1}^{K}\sum_t \log \pi_\theta(a_{i,t}\mid s_{i,t})\left(R_i-\frac{1}{K-1}\sum_{j\neq i}R_j\right) \\[6pt]
\textbf{CISPO:}\quad & -\sum_{i,t} \mathrm{sg}(\hat{\rho}_{i,t})\, A_{i,t} \log \pi_\theta(a_{i,t}\mid s_{i,t}) \\
& \quad \hat{\rho}_{i,t} = \mathrm{clip}(\rho_{i,t},\, 1-\varepsilon,\, 1+\varepsilon) \\[6pt]
\textbf{PPO:}\quad & -\frac{1}{T}\sum_{t=1}^{T}\min\!\big(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\, A_t\big) \\
& \quad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} \\[6pt]
\textbf{GRPO:}\quad & -\frac{1}{G}\sum_{i=1}^{G}\min\!\big(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)\, A_i\big) \\
& \quad \rho_i = \frac{\pi_\theta(a_i\mid s)}{\pi_{\theta_{\text{old}}}(a_i\mid s)},\quad A_i = \frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})} \\[6pt]
\textbf{GSPO:}\quad & -\frac{1}{G}\sum_{i=1}^{G}\min\!\big(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)\, A_i\big) \\
& \quad \rho_i = \left(\frac{\pi_\theta(a_i\mid s)}{\pi_{\theta_{\text{old}}}(a_i\mid s)}\right)^{1/|a_i|} \\[6pt]
\textbf{DPO:}\quad & -\mathbb{E}_{(x,y^{w},y^{l})}\!\left[\log \sigma\!\big(\beta[\Delta\log \pi_\theta(x)-\Delta\log \pi_{\mathrm{ref}}(x)]\big)\right]
\end{aligned}$$


## 实现

与这些算法最初被提出的深度 RL 文献相比，为优化语言模型或其他大型 AI 模型而实现 RL，需要许多细小的实现细节。
本节我们点出区分流行算法实现的一些关键因素。

这类训练还有许多其他小细节。
例如，用语言模型做 RLHF 时有一个关键步骤是生成供奖励模型打分的文本。
正常情况下，模型应当生成一个序列结束（EOS）token 来表示生成完毕；但常见做法是对生成长度设一个硬上限，以便高效利用基础设施。
RLHF 的一个失效模式是：模型的回答经常被截断，把奖励模型的评分推向分布之外、给出不可预测的分数。
解决办法是**只**对 `eos_token` 运行奖励模型打分，并对生成过长另外施加惩罚。

流行的开源 RLHF 工具在不同算法之间的实现细节差异很大（见 [@ivison2024unpacking] 的表 10）。
这里未覆盖的一些决策包括：

- **价值网络的初始化**：PPO 及类似算法内部使用的学到的价值网络，可以从同架构的另一个模型或随机权重开始。这对性能影响可能很大。InstructGPT 确立的标准做法 [@ouyang2022training]（并在 Tülu 3 的 RLVR 工作中沿用 [@lambert2024t]）是用 RLHF 所用的奖励模型来初始化价值网络。也有人用 RLHF 训练之前的检查点（通常是 SFT 模型）加上随机初始化的价值头，或者完全重新初始化的语言模型（较少见，因为 RLHF 收敛会更慢，但可行）。
- **奖励归一化、奖励白化，和/或优势白化**：归一化把奖励模型（或环境）给出的所有值约束到 0 与 1 之间，有助于学习稳定性。[白化](https://en.wikipedia.org/wiki/Whitening_transformation) 更进一步，把奖励或优势估计变换成零均值、单位方差，对稳定性的帮助更强。
- **不同的 KL 估计量**：对复杂语言模型，精确计算模型间的 KL 散度可能很复杂，因此会用多种近似来替代精确计算 [@schulman2016klapprox]。
- **KL 控制器**：PPO 及相关算法的原始实现带有动态控制器——它瞄准特定 KL 值，并根据近期测量结果改变惩罚强度。大多数现代 RLHF 实现使用静态 KL 惩罚，但这也会有所变化。

关于 RLHF 实现细节的更多内容，见 [@huang2024n]。
关于这些算法的更多信息，见 [@weng2018PG]。

### 策略梯度基础

一个简单的策略梯度实现——用优势来估计梯度，为 PPO、GRPO 这类高级算法做准备——如下：
```python
pg_loss = -advantages * ratio
```
这里的 ratio 是新策略模型概率相对于生成该批次的旧策略的（逐 token）概率比值（通常由对数概率之差算出）。

为理解这个式子，最好先理解一个更新批次里可能出现的几种情形。
记住：我们希望随着模型在任务上变好，损失**下降**。

情形 1：优势为正，说明该动作优于该状态的期望值。我们想强化它。这种情况下，负号会让模型试图提高该动作的可能性，做法是增大 logratio。正的 logratio——也就是这些 token 对数概率之和为正——意味着模型更可能生成这些 token。

情形 2：优势为负，说明该动作差于该状态的期望值。推理非常类似。此时如果新模型更可能产生该补全，损失就会是正的，于是模型会调整策略参数，让这个补全变得不那么可能。

情形 3：优势为零，无需更新。损失为零，不改变策略模型。

### 损失聚合的权衡

用语言模型实现任何策略梯度算法时都会遇到一个问题：**如何把逐 token 的损失聚合成最终的标量损失？**
给定样本 $i$ 在 token $t$ 处的逐 token 损失 $\ell_{i,t}$、补全长度 $|a_i|$ 与批大小 $B$，有三种主要策略：

**策略 1：按序列归一化**（GRPO 的标准做法；部分 PPO 实现也用）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_per_sequence}

每条序列对批次损失的贡献相等，与其长度无关。代码：

```python
# Strategy 1: Per-sequence normalization
sequence_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
             completion_mask.sum(dim=1)).mean()
```

**策略 2：按 token 归一化**（DAPO [@yu2025dapo]）

$$L = \frac{\sum_{i=1}^{B} \sum_{t=1}^{|a_i|} \ell_{i,t}}{\sum_{i=1}^{B} |a_i|}$$ {#eq:loss_per_token}

每个 token 贡献相等；更长的序列对梯度的影响按比例更大。代码：

```python
# Strategy 2: Per-token normalization
token_loss = ((per_token_loss * completion_mask).sum() / \
            completion_mask.sum())
```

**策略 3：固定长度归一化**（Dr. GRPO [@liu2025understanding]）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{L_{\max}} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_fixed_length}

按最大序列长度 $L_{\max}$ 归一化，在各序列之间拉平了逐 token 的尺度，同时仍让更长的序列贡献更多总梯度，因为它们包含更多有效 token。代码：

```python
# Strategy 3: Fixed-length normalization 
fixed_len_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
            L_max).mean()
```

其中 $L_{\max}$ 通常是整个训练过程中的全局常量，指定最大生成 token 数。

注意上面代码里的 `completion_mask` 是一个 0/1 矩阵，提示 token 被掩蔽（置 0），因为我们不想让模型从“预测提示 token”中学习。

#### 为什么这件事重要？

直观上，按序列归一化（策略 1）看起来最好，因为我们关心的是*结果*而不是单个 token。
然而它会基于序列长度引入微妙的偏置——视偏置方向不同，这可能让模型过度思考，或让那些天然需要更多 token 的策略权重被压低。
考虑两条长度不同、带逐 token 损失的序列：

```python
seq_1_losses = [1, 1, 1, 1, 10]  # 5 tokens, mean = 2.8
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 tokens, mean = 1.9
```

用**策略 1**（按序列）：批次损失为 $(2.8 + 1.9)/2 = 2.35$；关键在于，短序列中每个 token 获得的梯度比长序列中的更大。

用**策略 2**（按 token）：批次损失为 $(14 + 19)/15 = 2.2$，所有 token 获得相同量级的梯度。

用**策略 3**（固定长度，$L_{\max}=10$）：短序列贡献 $1.4$，长序列贡献 $1.9$，在按序列加权的同时又拉平了逐 token 梯度。

一个更完整、展示这些策略如何影响梯度的例子见下面的脚本。

```python
from typing import Optional
import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[int] = None,
        constant_normalizer: float = 1.0,
    ) -> torch.Tensor:
    """Compute sum of tensor with masked values. Use a constant to normalize."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / constant_normalizer
    else:
        return (values * mask).sum() / constant_normalizer

ratio = torch.tensor([
    [1., 1, 1, 1, 1, 1, 1,],
    [1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)


advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2,],
    [2, 2, 2, 2, 2, 2, 2,],
])

masks = torch.tensor([
    # generation 1: 4 tokens
    [1, 1, 1, 1, 0, 0, 0,],
    # generation 2: 7 tokens
    [1, 1, 1, 1, 1, 1, 1,],
])

max_gen_len = 7

masked_mean_result = masked_mean(ratio * advs, masks, axis=1)
masked_mean_token_level = masked_mean(ratio, masks, axis=None)
masked_sum_result = masked_sum(ratio * advs, masks, axis=1, constant_normalizer=max_gen_len)

print("masked_mean", masked_mean_result)
print("masked_sum", masked_sum_result)
print("masked_mean_token_level", masked_mean_token_level)

# masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
# masked_sum tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
# masked_mean_token_level tensor(1., grad_fn=<DivBackward0>)

masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_sum_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_mean_token_level.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad tensor([[0.0909, 0.0909, 0.0909, 0.0909, 0.0000, 0.0000, 0.0000],
# [0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909, 0.0909]])
```

输出显示：用策略 1（`masked_mean`）时，短序列的逐 token 梯度（0.25）大于长序列（0.14）。
策略 2 与策略 3 则拉平了各序列的逐 token 梯度。
注意如果使用梯度累积（在回传前把多个小批次的梯度相加），这些结果可能显著变化——那种情况下，短序列与长序列之间的平衡可能反转。

实践中，最佳策略取决于具体训练设置。
在 RLHF 中，往往偏好数值稳定性最好、或损失方差最小的那种。

#### 相关话题：MDP 与赌博机框架

损失聚合的选择，与“我们如何框定 RL 问题”这一更深的区分相关。
**MDP（token 级）**视角把每个 token $a_t$ 当作一个动作，状态 $s_t$ 是到目前为止的前缀。
实践中，这是我们在用学到的价值函数 $V(s_t)$（例如 GAE [@schulman2015high]）计算逐 token 优势、并逐 token 施加 KL 惩罚时所用的框架。
带学到的价值网络的 PPO 是经典例子 [@schulman2017proximal]。

相比之下，**赌博机（序列级）**视角把整个补全当作一个动作、配一个标量奖励 $R$。
在代码中，这意味着计算一个序列级优势 $A_{\text{seq}}$ 并把它广播到所有 token。
RLOO 与 GRPO 式优势常用于这种赌博机式设定 [@kool2019buy] [@ahmadian2024back] [@shao2024deepseekmath]。
DPO 与 A-LoL 这类直接对齐方法也定义了序列级目标，尽管它们不是策略梯度估计量 [@baheti2023leftover]。

注意许多 GRPO 实现同时使用赌博机式优势**和**在损失中另加一个逐 token 的 KL 项，而许多 PPO/RLOO 实现则在计算优势之前把 KL 折进奖励——两种约定在实践中都存在。

下面是一个突出两种做法的对比示例：

```python
# === Bandit-style (sequence-level) ===
# One scalar reward per sequence; advantage broadcast to all tokens
reward = torch.tensor([3.0, 1.0])       # (B,) e.g., reward model scores
baseline = reward.mean()                 # simple baseline (RLOO uses leave-one-out)
advantage_seq = reward - baseline        # (B,)
advantages = advantage_seq[:, None].expand(-1, seq_len)  # (B, L)
# tensor([[ 1.,  1.,  1.,  1.],    <- same advantage for all tokens
#         [-1., -1., -1., -1.]])

# === MDP-style (token-level) ===
# Per-token rewards + learned V(s_t); each token gets its own advantage
# (could also use per-token KL shaping, format rewards, or other token-level signals)
advantages = gae(per_token_rewards, values, done_mask, gamma=1.0, lam=0.95)
# tensor([[ 0.2,  0.5,  0.8,  1.5],    <- varies by position
#         [-0.3, -0.5, -0.8, -1.4]])
```

这个框架区分也解释了：为什么几乎所有 RLHF 实现都把折扣因子 $\gamma$ 设为 1.0。
在标准 RL 中，折扣（$\gamma < 1$）是必需的：它在多步回合中平衡短期与长期奖励，对智能体学会长期有效行为至关重要。
但在 RLHF 设定中，即便采用 token 级 MDP 视角，优化的归纳偏置仍是“整个补全的质量”——奖励信号给的是整个回复打分，而非单个 token。
对更早的 token 打折，只会毫无原则地压低它们的贡献。
随着智能体式 RL 设定走向成熟——模型在那里采取真正的多步动作，如工具调用、执行代码、浏览网页——折扣可能重新变得相关，因为这些确实涉及彼此不同、长远后果各异的序列决策。

### 异步 RL 系统

策略梯度算法的默认实现是所谓的**同策略**执行：智能体（语言模型）采取的动作（生成内容）在被评分之后才用于更新模型。
策略梯度的理论推导依赖所有动作都严格同策略——模型始终与最新试验/rollout 的结果保持同步。
实践中，维持严格的同策略执行会大幅拖慢训练 [@noukhovitch2024asynchronous]——而且完美同步在技术上本就不可能。
因此，近期所有语言模型的经验结果，往往都略微落在理论证明之外。
实践中真正发生的，是**为“实际有效”而设计算法与系统**。

![遵循 Noukhovitch 等 2024 的同步与异步 RL 训练中“生成—更新”阶段对比。](images/async_v_synch_rl.png){#fig:async}

常用的解决方案是：持续在**分离的 GPU 节点**上分别运行推理与训练，并用软件让两者高效并行，如 @fig:async 底部所示。
流行开源语言模型 RL 工具的常见做法，是用 Ray 这类分布式进程管理库，在策略梯度学习循环与推理循环之间传递信息，后者使用高效的推理引擎（如 vLLM）。
这类设置中，专门做 RL 步的 GPU 称为“learner”，专门从语言模型采样的 GPU 称为“actor”。
让训练更加异步时面临的主要挑战，是**保持训练稳定并维持学习信号**。

![一个分布式 RL 系统的例子：用两个队列把数据传递给 learner 与 actor GPU，两者都可以通过 Ray 这类分布式计算库同步。Olmo 团队 2025，许可 CC-BY。](images/distributed-rl.png){#fig:async_system}

这类系统的设计与实现前提是：**近似同策略的数据对稳定学习已经足够好。**
在这里，生成阶段与更新阶段可以很方便地同步，以避免训练系统任一侧的空闲算力——也就是 @fig:async_system 中把模型权重从 learner 传给 actor。
对推理模型而言，那些需要每个答案 1 万到 10 万以上 token 的问题，其超长推理特性让 rollout 生成成为更强的瓶颈。
在更同步的 RL 基础设施上训练推理模型时，一个常见问题是：批次中某个提示的答案可能要多得多的时间才能生成完（无论是更多 token 还是更多工具调用），导致已分配的大部分算力一直空转到它完成为止。
针对这种长度不匹配的第二个方案叫**序列级打包**（sequence-level packing）：把批次中较短的样本用巧妙的掩码堆叠起来，使模型能持续 rollout，并在批次内更均衡地分摊长度归一化。
分布式 RL 基础设施的完整复杂度超出本书范围——它还会引发许多其他微妙问题，拖慢训练或造成不稳定。

这些推理模型出现之后，人们对“让训练与推理循环完全异策略”产生了更多兴趣：策略梯度更新的训练批次，由多个生成实例上最近完成的 rollout 填充 [@wu2025llamarl] [@fu2025areal]。
完全异步训练还能让 RL 训练更容易跨多个数据中心扩展——因为可以选择拉长 learner 节点（做策略梯度步）与 actor（解题）之间权重同步的间隔 [@primeintellectteam2025intellect2reasoningmodeltrained]。

相关工作正在探索完全异策略的策略梯度算法 [@leroux2025topr]。

### 截断重要性采样

截断重要性采样（Truncated Importance Sampling，TIS）是用于稳定现代异步语言模型 RL 框架训练的关键工具。
重要性采样是一种修正：对来自某个分布的样本重新加权，以估计另一个分布下的期望（如 @eq:IS_identity 所介绍）。
截断重要性采样 [@ionides2008truncated] 用 $\min(\rho, C)$（$C$ 为某常数）给这些权重封顶，用一点偏差换取策略梯度中有界的方差。

这是施加在策略梯度上的重要性采样修正；但与 PPO、CISPO 的双边裁剪（把比值约束在 1 附近）不同，TIS 使用**单侧上界**：比值可以自由地低于 1，但被 $C$ 封顶以防止极端的增权。
在 PPO、GRPO、CISPO（及相关算法）中，比值 $\rho_t^{\text{policy}} = \pi_\theta(a_t \mid s) / \pi_{\theta_{\text{old}}}(a_t \mid s)$ 修正的是“同一 RL 批次内多个梯度步”造成的策略漂移。
当我们转向真实世界的 RL 框架——以上一小节的异步性为核心——还会出现更大的数值差异来源（同样需要重要性采样的数值修正）。
即便采样器与 learner 共享完全相同的参数 $\theta$，它们的有效 token 分布也可能不同，因为推理引擎（如 vLLM）与训练框架（如 FSDP）使用不同的 kernel、精度和并行策略 [@yao2025offpolicy]。
因此有必要区分“同一个策略在两个系统上”的求值 $\pi_\theta^{\text{sampler}}$ 与 $\pi_\theta^{\text{learner}}$，并定义相应的比值及其截断形式：

$$
\rho_t^{\text{learner}} = \frac{\pi_\theta^{\text{learner}}(a_t \mid s, a_{<t})}{\pi_\theta^{\text{sampler}}(a_t \mid s, a_{<t})}, \qquad \tilde{\rho}_t^{\text{learner}} = \min(\rho_t^{\text{learner}},\; C).
$$ {#eq:tis_backend}

这两种修正互补，但出现在策略梯度实现中的原因不同——一个补偿同一 RL 批次训练内部的策略漂移，另一个补偿实现引入的偏差——两者可以同时施加。
它们如何结合取决于具体算法：

#### 带 TIS 的 REINFORCE（单个梯度步）

此时不存在策略漂移（$\pi_\theta = \pi_{\theta_\text{old}}$），唯一的失配来自 learner 与 sampler 之间。
这里 $\pi_{\theta_\text{old}} = \pi_\text{gen}$，TIS 直接修正 learner–sampler 的差距：

$$
\nabla_\theta J \approx \mathbb{E}_{a \sim \pi_\theta^{\text{sampler}}} \left[ \tilde{\rho}_t^{\text{learner}} \cdot A_t \cdot \nabla_\theta \log \pi_\theta^{\text{learner}}(a_t \mid s, a_{<t}) \right].
$$ {#eq:reinforce_tis}

#### 带 TIS 的 PPO/GRPO（多个梯度步）

此时两个比值都在起作用。
在严谨的实现中，策略比值里的“旧对数概率”会在 learner 上重新计算（GSPO 论文讨论了这一点），因此策略比值 $\rho_t^{\text{policy}} = \pi_\theta^{\text{learner}} / \pi_{\theta_\text{old}}^{\text{learner}}$ 捕捉的是纯粹的策略漂移，而 $\tilde{\rho}_t^{\text{learner}} = \min(\pi_{\theta_\text{old}}^{\text{learner}} / \pi_{\theta_\text{old}}^{\text{sampler}},\; C)$ 单独修正在生成检查点处的后端失配：

$$
J_{\text{PPO+TIS}}(\theta) = \mathbb{E}\left[ \min\!\left( \rho_t^{\text{policy}}\, A_t,\; \text{clip}\!\left(\rho_t^{\text{policy}}, 1-\varepsilon, 1+\varepsilon\right) A_t \right) \cdot \tilde{\rho}_t^{\text{learner}} \right].
$$ {#eq:ppo_tis}

这里 $\pi_{\theta_\text{old}} \neq \pi_\text{gen}$：旧对数概率来自 learner，而不是 sampler。
如果某个框架跳过这次重算、直接把 sampler 的对数概率当作 $\pi_{\theta_\text{old}}$，那么策略比值本身就已经包含了后端失配，也就不需要单独的 TIS 修正——但此时裁剪作用在一个更嘈杂的比值上，它在任何梯度步之前就已经偏离 1.0。
这正是 Yao 等 [-@yao2025offpolicy] 所说“你的框架悄悄给你带来了异策略 RL”的那一点。

实践中，LLM RL 系统把 TIS 作为策略梯度损失上的逐 token 修正权重来施加：

```python
# Shape: (B*G, L)
C = 2.0  # TIS cap

logratio = learner_logprobs - sampler_logprobs
logratio = logratio.clamp(-10.0, 10.0)              # numerical safety
tis_weight = torch.exp(logratio).clamp(max=C)        # one-sided truncation

# Use as a fixed correction weight on the per-token PG loss
per_token_pg_loss = per_token_pg_loss * tis_weight.detach()
```

$[-10, 10]$ 的 clamp 只是为了取指数前的数值稳定；真正执行截断重要性采样的是在 $C$ 处的单侧封顶。
实践中，围绕这些对数概率的记账工作——保存生成时的 sampler 对数概率、在旧检查点处重算 learner 对数概率、在梯度步中跟踪当前对数概率——构成了分布式 RL 框架脚手架中相当大的一部分。
与 GSPO 不同，这个修正是 token 级的，因为它处理的是 token 级的数值失配，而非序列级的奖励粒度。
针对 learner–sampler 比值的 TIS 已被主要开源 RL 框架采用（VeRL、TRL、OpenRLHF、SkyRL、OAT，以及使用 $C = 2$ 的 Open Instruct），并且对长推理轨迹（第 7 章）越来越重要——在那里，逐 token 的微小差异会在数千个生成 token 上不断累积。


### 示例：PPO

PPO 有非常多的实现。
核心的*损失*计算如下。
而稳定性能的关键还在于*价值*的计算，那里存在多种选择（包括*价值模型*损失本身的多种选择）。

注意这里的参考策略（或旧对数概率）来自生成被采样之时，**不一定**是参考模型。
参考模型只用于 KL 距离约束/惩罚。

```python
# B: Batch Size, L: Sequence Length, G: Num of Generations
# Apply KL penalty to rewards
rewards = rewards - self.beta * per_token_kl  # Shape: (B*G, L)

# Get value predictions
values = value_net(completions)  # Shape: (B*G, L)

# Compute returns via backward pass (gamma typically 1.0 for LM RLHF)
# Mask rewards to avoid padding tokens (which may have KL penalties) leaking into returns
returns = torch.zeros_like(rewards)
running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(rewards.shape[1])):
    # Zero out padding: only accumulate rewards/returns for valid completion tokens
    running = (rewards[:, t] + self.gamma * running) * completion_mask[:, t]
    returns[:, t] = running

# Compute advantages: A_t = G_t - V(s_t)
advantages = returns - values.detach()  # Shape: (B*G, L)
# Note: We detach the value network here to not update the parameters of
# the value function when computing the policy-gradient loss

# Normalize advantages (optional but stable)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# Value function loss: predict returns
vf_loss = 0.5 * ((returns - values) ** 2)  # Shape: (B*G, L)

# Combine policy and value losses
per_token_loss = pg_loss_max + self.vf_coef * vf_loss  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute metrics for logging
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()
    
    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()
    
    # Compute value loss for logging
    value_loss = vf_loss.mean()
```

理解 PPO 的核心，是弄清策略梯度损失是如何更新的。
重点看这三行：
```python
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)
```
`pg_losses1` 是朴素的优势加权策略梯度损失。`pg_losses2` 用同一公式，但把概率比值钳制到 $[1-\varepsilon, 1+\varepsilon]$，限制策略在单次更新中能改变多少。

关键洞见在于对两个损失取 `torch.max`。因为我们是在最小化一个*负*损失（回忆一下优势前面的负号），取最大值就是选择更悲观的梯度——也就是产生更小策略更新的那个。当优势为正（好动作）时，裁剪防止策略过于激进地提高该动作的概率；当优势为负（坏动作）时，裁剪防止反方向的过度纠正。

通过钳制对数概率比值，PPO 给“策略相对生成训练数据那一版能漂移多远”设了界，从而在不需要显式计算信任域的前提下稳定学习。

上面的代码还展示了 PPO 会与策略同步学习一个价值函数，这增加了实现复杂度，但裁剪目标才是核心机制。

#### 每样本单梯度步时 PPO/GRPO 的简化（无裁剪）

如果超参“每样本梯度步数”等于 1，PPO（与 GRPO）的实现可以优雅得多。
这个超参的典型取值常在 2–4 或更高。
在 PPO 或 GRPO 的主方程（见 @eq:PPO_EQN）中，“参考”策略就是先前的参数——即用于生成补全或动作的那一版。
因此，如果只做一次梯度步，就有 $\pi_\theta = \pi_{\theta_{\text{old}}}$，更新规则化简为下式（记号 $[]_\nabla$ 表示 stop-gradient）：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\frac{\pi_\theta(a_i|s)}{\left[\pi_{\theta}(a_i|s)\right]_\nabla}A_i - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right). $$ {#eq:ppo_1step}

这就带来了可以省略第二项策略梯度与裁剪逻辑的 PPO 或 GRPO 实现，让优化器更接近标准策略梯度。


### 示例：GRPO

DeepSeekMath 论文描述了 GRPO 一些与 PPO 不同的实现细节 [@shao2024deepseekmath]，尤其是在与“深度 RL 中标准 PPO 用法”而非语言模型场景对比时。
例如，RLHF 优化中的 KL 惩罚（回忆一下：在没有奖励模型的、用可验证奖励训练推理模型时也会用到 KL 惩罚）是直接施加在损失更新里，而不是加在奖励函数上。
RLHF 中标准的 KL 惩罚用法是 $r=r_\theta - \beta \mathcal{D}_{\text{KL}}$，而 GRPO 实现大致是这样的：

$$ L = L_{\text{policy gradient}} + \beta * \mathcal{D}_{\text{KL}} $$ {#eq:grpo_loss_kl}

不过实现方式有多种。
传统上，KL 距离是针对“对提示 $s$ 的补全”中每个 token 计算的。
对推理训练，一个提示会采样多个补全，而一个批次里有多个提示，
所以 KL 距离的形状会是 [B, L, N]，其中 B 是批大小、L 是序列长度、N 是每个提示的补全数。

合起来，用第一种损失累加方式，伪代码可以写成下面这样。

```python
# B: Batch Size, L: Sequence Length, G: Number of Generations
# Compute group-wise rewards # Shape: (B,)
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)    


# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
# Shape: (B*G,)

# Compute advantages
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
advantages = advantages.unsqueeze(1)
# Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# important to GRPO -- PPO applies this in reward traditionally
# Combine with KL penalty
per_token_loss = pg_loss_max + self.beta * per_token_kl  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute core metric for logging (KL, reward, etc. also logged)
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()
    
    # Compute approximate KL
    approx_kl = (0.5 * ((new_per_token_logps - per_token_logps)**2) * completion_mask).sum() / completion_mask.sum()
```

关于如何解读这段代码的更多细节，见上面的 PPO 一节。与 PPO 示例的核心差别是：

- **优势计算**：GRPO 把奖励相对于组内（同一提示的多次生成之间的均值与标准差）做归一化，而不是用学到的价值函数当 baseline。
- **没有价值网络**：GRPO 完全去掉了价值模型，消除了 `vf_loss` 以及与之相关的复杂度。
- **KL 惩罚的位置**：GRPO 把 KL 惩罚直接加在损失里，而不是从奖励中减去（这是标准实现，但关于如何施加 KL 还有更多版本）。

#### RLOO 与 GRPO

RLOO 的优势更新与 GRPO 非常接近，这凸显了：一旦把 PPO 式裁剪与 KL 惩罚的细节单独拿出来，两个算法在概念上何其相似。
具体来说，对 RLOO，优势是相对于一个与 GRPO 极为相似的 baseline 计算的——都是“该补全相对于同一问题下其他补全”的奖励。
简洁地说，RLOO 的优势估计如下（展开自 [TRL](https://github.com/huggingface/trl/blob/bfe20756082488350091352d1cdc19c172e42cd8/trl/trainer/rloo_trainer.py#L433) 的实现）：

```python
# rloo_k --> number of completions per prompt 
# rlhf_reward --> Initially a flat tensor of total rewards for all completions. Length B = N x k
rlhf_reward = rlhf_reward.reshape(rloo_k, -1) # 
# Now, Shape: (k, N), each column j contains the k rewards for prompt j.

baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
# baseline --> Leave-one-out baseline rewards. Shape: (k, N)
#  baseline[i, j] is the avg reward of samples i' != i for prompt j.

advantages = rlhf_reward - baseline
# advantages --> Same Shape: (k, N)

advantages = advantages.flatten() # Same shape as original tensor
```

RLOO 其余的实现细节，沿用实现策略梯度时的其他权衡。

## 补充话题

要精通策略梯度算法的应用，还有无数其他考量。
这里我们讨论成功部署策略梯度 RL 算法时的一些长尾复杂问题。

### 广义优势估计（GAE）

广义优势估计（GAE）是策略梯度算法中计算优势的另一种方法 [@schulman2015high]，它能更好地平衡偏差-方差权衡。
传统的单步优势估计可能引入过多偏差，而使用完整轨迹又可能方差过大。
GAE 计算多步优势估计的指数加权平均，其中 $\lambda$ 超参控制偏差-方差权衡——从单步 TD（$\lambda=0$）到完整轨迹回报（$\lambda=1$）；$\lambda=0.95$ 是 LLM 微调的常见默认值。

优势估计可以有很多形式，但我们可以把 $n$ 步优势估计量定义为如下（类似本章开头的 TD 残差）：

$$
\hat{A}_t^{(n)} = \begin{cases}
r_t + \gamma V(s_{t+1}) - V(s_t), & n = 1 \\
r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t), & n = 2 \\
\vdots \\
r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots - V(s_t), & n = \infty
\end{cases}
$$ {#eq:K_STEP_ADV}

这里更短的 $n$ 意味着更低的方差但更高的偏差——因为我们把更多的学习能力归因到每条轨迹上，它可能过拟合。
GAE 试图把这一表述推广为加权多步平均，而不是取某个具体的 $n$。
首先，我们必须定义预测价值的时序差分（TD）残差。

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$ {#eq:TD_RESIDUAL}

为利用它，我们引入另一个变量 $\lambda$ 作为 GAE 的混合参数。它折叠为对未来优势的指数衰减：

$$
\begin{array}{l}
\hat{A}_t^{GAE(\gamma,\lambda)} = (1-\lambda)(\hat{A}_t^{(1)} + \lambda\hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + \cdots) \\
= (1-\lambda)(\delta_t^V + \lambda(\delta_t^V + \gamma\delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V) + \cdots) \\
= (1-\lambda)(\delta_t^V(1 + \lambda + \lambda^2 + \cdots) + \gamma\delta_{t+1}^V(\lambda + \lambda^2 + \cdots) + \cdots) \\
= (1-\lambda)\left(\delta_t^V\frac{1}{1-\lambda} + \gamma\delta_{t+1}^V\frac{\lambda}{1-\lambda} + \cdots\right) \\
= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V
\end{array}
$$ {#eq:GAE_DFN}

直观上，这以优雅的方式对优势的多步估计做了平均。
一个示例实现如下：

```python
# GAE (token-level) for LM RLHF
#
# B: Batch Size
# L: Length
# Inputs:
#   rewards: (B, L) post-KL per-token rewards
#   values:  (B, L) current V_theta(s_t)
#   done_mask: (B, L) 1.0 at terminal token (EOS or penalized trunc), else 0.0
#   gamma: float (often 1.0), 
#   lam (short for lambda): float in [0,1]
#   (Padding beyond terminal should have rewards=0, values=0)
B, L = rewards.shape
advantages = torch.zeros_like(rewards)
next_v = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

for t in reversed(range(L)):
    not_done = 1.0 - done_mask[:, t]
    delta = rewards[:, t] + gamma * not_done * next_v - values[:, t]
    gae = delta + gamma * lam * not_done * gae
    advantages[:, t] = gae
    next_v = values[:, t]

targets = advantages + values      # y_t for value regression
advantages = advantages.detach()   # for policy loss
```

这个反向循环累加时序差分（TD）误差（$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$），它衡量实际结果比价值函数的预测好多少或差多少，并带指数衰减 $(\gamma\lambda)^l$。
在终止 token 处，`not_done=0` 阻止从未来状态做自举，并重置 GAE 累加器，因此每个回合的优势是独立计算的（由于循环反向运行，终止 token 会干净地在回合边界处停止指数加权累加——这让实现天然兼容序列打包，能正确处理拼接进一条序列的多个样本）。
最后的 `targets` 用作在本 GAE 循环之外学习的价值函数的回归目标，而分离出来的 `advantages` 为策略梯度加权——分离是为了让策略更新不通过价值网络回传。
在语言模型的 RLHF 中，$\gamma=1.0$ 很常见，因为回合是简短的 token 序列，此时更偏好不折扣的信用分配（而且常常一条序列里全部 token 都是同一个回合）。

*延伸阅读见 [@seita2017gae]。*

### 双重正则化

本章我们已经看到两类正则化。一类内建于 PPO 这类算法中，约束步长；另一类是针对优化起点的、基于 KL 散度的距离惩罚。

深度强化学习中许多流行的策略梯度算法——包括 PPO 及其前身——之所以出现，就是因为需要控制智能体的学习过程。
而在 RLHF 中，如第 15 章「正则化」与第 3 章「训练总览」所详述，由于存在相对于“被微调的原始策略”的距离惩罚，正则化是**内建**的。
从这个视角看，PPO（带有内部步长正则）与 REINFORCE（更简单，在某些超参下 PPO 会退化到它）这类算法之间的差异，对**微调语言模型**而言远不如“从零训练一个智能体”那样有意义。

PPO 中负责给更新步长封顶的目标被称为[代理目标](https://huggingface.co/blog/deep-rl-ppo#introducing-the-clipped-surrogate-objective)。
要监控 PPO 正则化对 RLHF 更新的影响程度，可以看许多流行实现中的 clip fraction 变量——即批次中概率比值落在裁剪区间之外的样本占比。
它是“PPO 正则化何时可能生效”的有用代理指标，但并非每个这样的样本梯度都为零：只有当裁剪分支被选中时（例如正优势且比值高于 $1+\varepsilon$，或负优势且比值低于 $1-\varepsilon$），代理目标才会变平。

在语言模型实践中，PPO、GRPO 这类算法常常每批只做**一个**梯度步，这意味着 PPO 原生的正则化从未被触发（裁剪只可能在策略发生明显变化时于批次内出现），此时是 KL 距离惩罚在起主导作用。
不过这并不普遍。例如 DAPO 每批用 16 个梯度步 [@yu2025dapo]；Tülu 3 对 8B 和 70B 模型每批用 4 次 PPO 更新迭代，但对 405B 降到 1 次以维持训练稳定 [@lambert2024t]。

### 延伸阅读

随着 RLHF 稳固地居于现代后训练的中心，人们提出了其他策略梯度 RL 算法以及更广义的 RL 算法来改进训练过程，但它们并未在最佳实践的制定中占据核心地位。
延伸阅读的例子包括：

- **成对近端策略优化（P3O；Wu 等，2023）** [@wu2023pairwise] 直接在 PPO 式策略更新中使用成对数据，而不学习中间的奖励模型。
- **软自适应策略优化（SAPO）** [@gao2025sapo] 用平滑的、温度控制的门控取代 PPO/GRPO 式的硬裁剪，目标是形成一个连续的信任域——在压低异策略 token 权重的同时，保留近似同策略的学习信号。
- 异策略策略梯度算法可能支持更进一步的异步训练，例如 **对比策略梯度（CoPG）** [@flet2024contrastive]（直接对齐算法 IPO 与朴素策略梯度的推广），Cohere 在其 Command A 模型上使用了它 [@cohere2025command]。
- 还有其他为语言模型设计的 REINFORCE 实现，例如 **ReMax** [@li2023remax]，它实现了一种专门为吸收“奖励模型推理带来的不确定性”而设计的 baseline 归一化。
- 一些基础模型——例如 Apple Intelligence Foundation Models [@gunter2024apple] 或 Kimi k1.5 推理模型 [@team2025kimi]——使用了 **镜像下降策略优化（MDPO）** [@tomar2020mirror] 的变体。这方面的基础研究仍在发展 [@zhang2025improving]，但镜像下降是一种优化方法，而非直接的策略梯度算法。这里重要的是，它被嵌入的方式与既有 RL 基础设施非常相似。
- **解耦裁剪与动态采样策略优化（DAPO）** 对 GRPO 提出四项改动，以更好适配推理语言模型——那里需要长轨迹，也需要提高那些新颖、利用不足的 token 的概率 [@yu2025dapo]。改动是：1）使用两个不同的裁剪超参 $\varepsilon_\text{low}$ 与 $\varepsilon_\text{high}$，使对数比值正侧的裁剪可以迈出更大步子以更好地探索；2）动态采样——去掉批次中奖励全为 0 或全为 1 的样本（没有学习信号）；3）使用前面「实现：GRPO」中讨论的逐 token 损失；4）对过长的样本施加软惩罚，避免从被截断的答案中学习。
- **基于价值的增强近端策略优化（VAPO）** [@yuan2025vapo] 把 DAPO 的若干优化（包括 clip-higher、token 级策略梯度、不同的长度归一化）与 Value-Calibrated PPO [@yuan2025s] 的洞见结合起来——预训练价值函数、长度自适应 GAE——展示了相对于 GRPO，基于价值的方法仍具潜力。

## 建议的实验

`code/policy_gradients/` 中的配套实现，是为“小型、可观察的 RL 运行”设计的。
默认配置在 `reasoning-gym` 的 `spell_backward` 程序化任务上训练 `Qwen/Qwen3-1.7B`；这是一个很好的入门练习，因为失败与部分进展都容易观察。

1. **用 GRPO 跑词序反转任务。**

   ```bash
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

   跟踪 `avg_correctness`、`avg_format` 和 `avg_binary`。
   第一个有用的问题是：每个提示组内部是否**存在对比**——如果所有采样补全全对或全错，组相对更新就几乎没有学习信号。

2. **比较组相对估计量与单样本估计量。**
   运行配对好的起始配置：

   ```bash
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/reinforce.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/rloo.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

   比较正确率信号提升的速度，以及损失有多吵。
   RLOO 与 GRPO 会让“提示内 baseline”的作用比单看方程具体得多。

3. **扫对比度相关的旋钮。**
   复制 `policy_gradients/configs/grpo.yaml`，改变 `num_rollouts`、`temperature`、`data.size` 和 `format_weight`。
   较小的 `num_rollouts` 会降低组内对比度；温度过低会让采样坍缩；温度过高会生成过多格式错乱的答案。
   这是理解“为什么 RLVR 配方常在动优化器之前，先在采样设置上花很大功夫”的最简单方式。

4. **从玩具奖励走向数学。**
   对于 GSM8K 类实验，先看 `code/reward_models/train_orm.py` 与 `code/rejection_sampling/` 的例子，再考虑新增在线 RL 环境。
   一个有价值的贡献是：写一个能在 1B 以下的 Qwen 模型上运行的小型 `reasoning-gym` 或 GSM8K 策略梯度配置，并报告同样的组内对比度诊断指标。
