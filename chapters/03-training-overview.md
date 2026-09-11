<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
-->
---
prev-chapter: "RLHF 简史"
prev-url: "02-related-works"
page-title: 训练总览
search-title: "第 3 章：训练总览"
meta-description: "现代后训练流程的高层地图，涵盖指令微调、RLHF、RLVR 与直接对齐方法。"
next-chapter: "指令微调"
next-url: "04-instruction-tuning"
lectures:
  - video: "https://www.youtube.com/watch?v=MMDNaeIFVy8&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "第 0 讲：前置知识"
  - video: "https://www.youtube.com/watch?v=o6l6tJQgUg4&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "第 1 讲：总览（第 1–3 章）"
---

# 训练总览

本章先对 RLHF 训练做一个粗线条的总览，具体细节留到本书后面展开。
RLHF 虽然只优化一个简单的损失函数，却要按顺序训练多个不同的 AI 模型，再把它们连接成一个复杂的在线优化过程。

这里我们先介绍 RLHF 的核心目标：在带距离正则项的前提下，优化一个关于人类偏好的代理奖励（并说明它与经典 RL 问题的关系）。
然后我们会展示一些经典流程，说明用 RLHF 打造领先模型时，它如何与后训练的其他方法衔接。
这些示例流程会成为本书后续内容的参照——后面我们会讲做 RLHF 时可选择的不同优化方案，并回头指出不同关键模型在训练中各用了哪些步骤。

## 问题形式化

人类反馈强化学习（RLHF）的优化，建立在标准 RL 设定之上。
在 RL 中，智能体在给定环境状态 $s_t$ 时，从策略 $\pi(a_t\mid s_t)$ 中采样动作 $a_t$，以最大化奖励 $r(s_t,a_t)$ [@sutton2018reinforcement]。
策略是一个把每个状态映射到动作概率分布的函数。
演化为现代 RLHF 文献的早期策略，属于所谓的深度强化学习——即用神经网络来学习上述函数。
传统上，环境按转移（动力学）$p(s_{t+1}\mid s_t, a_t)$ 演化，初始状态服从分布 $\rho_0(s_0)$。
策略与动力学共同诱导出一个轨迹分布。
一条轨迹的总体概率，等于初始状态概率、策略做出的每一次动作选择、以及环境产生的每一次状态转移三者的乘积：

$$p_{\pi}(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t).$$ {#eq:rl_dynam}

在一个长度为 $T$ 的有限回合（episode）中，RL 智能体的目标是求解如下优化问题，其中 $\gamma$ 是 0 到 1 之间的折扣因子，用来权衡近期奖励与远期奖励的相对可取性：

$$\max_\pi \; \mathbb{E}_{\tau \sim p_{\pi}} \left[ \sum_{t=0}^{T-1} \gamma^t r(s_t, a_t) \right].$$ {#eq:rl_opt}

给定策略的期望回报通常记作 $J(\pi)$，其最优值记作 $J^* = \max_\pi J(\pi)$。

对于持续性任务，通常取 $T\to\infty$，并依靠折扣（$\gamma<1$）使目标函数保持良定义。
优化该表达式的多种方法将在第 6 章讨论。

![标准 RL 循环](images/rl.png){#fig:rl width=320px .center data-dark-src=“images/rl-dark.png”}

RL 循环的一个标准图示见 @fig:rl（可与 @fig:rlhf 中的 RLHF 循环对照）。

### 一个简单例子：恒温器 {#example-rl-thermostat}

为了对 RL 在做什么建立基本直觉，设想一个恒温器试图把房间维持在 70$^\circ$F 的目标温度。
在 RL 中，智能体一开始对任务一无所知，必须通过试错发现一个好策略。
恒温器这个例子包含以下组件（各部分如何对应 @eq:rl_dynam 中的轨迹分布，见 @fig:thermostat-equation）：

- **状态（$s_t$）**：当前室温，例如 65$^\circ$F。
- **动作（$a_t$）**：打开或关闭加热器。
- **奖励（$r$）**：温度与目标相差 2$^\circ$ 以内时为 +1，否则为 0。
- **策略（$\pi$）**：根据当前温度决定开或关加热器的规则。下面是恒温器可能学到的一个策略——依环境的具体转移动力学不同，它未必最优：

$$\pi(a_t = \text{on} \mid s_t) = \begin{cases} 1 & \text{if } s_t < 70^{\circ}\text{F} \\ 0 & \text{otherwise} \end{cases}$$ {#eq:thermostat_policy}

- **转移**：加热器打开时房间升温，关闭时降温。智能体通过动作影响这些动力学，但底层物理——房间升温或降温有多快——不在它的控制之内。

![轨迹分布（@eq:rl_dynam）中的每一项与恒温器 RL 示例的对应关系。](images/thermostat_equation.png){#fig:thermostat-equation .center data-dark-src=“images/thermostat_equation-dark.png”}

起初，恒温器的策略基本是随机的——它不管当前温度就胡乱开关加热器，房间温度剧烈波动。
经过许多回合的试错，智能体发现：房间冷时开加热器、热时关掉，能带来更多奖励，于是逐渐收敛到一个合理的策略。
这就是 RL 的核心循环：观察状态、选择动作、获得奖励、更新策略，以便随时间获得更多奖励。

### 经典 RL 示例：CartPole

要看一个动力学连续、内容更丰富的例子，可以考虑经典的 *CartPole*（倒立摆）控制任务——它出现在许多 RL 教材、课程甚至研究论文中。
恒温器只有一个状态变量和二元动作，而 CartPole 涉及四个连续状态变量和基于物理的转移，因此成为 RL 算法的标准基准。

![CartPole 环境，展示状态变量（$x$、$\dot{x}$、$\theta$、$\dot{\theta}$）与动作（$\pm F$）。](images/cartpole.png){#fig:cartpole width=400px .center data-dark-src=“images/cartpole-dark.png”}

- **状态（$s_t$）**：小车位置/速度与杆的角度/角速度：

  $$s_t = (x_t,\,\dot{x}_t,\,\theta_t,\,\dot{\theta}_t).$$ {#eq:cartpole_state}

- **动作（$a_t$）**：对小车施加左/右方向的水平力，例如 $a_t \in \{-F, +F\}$。

- **奖励（$r$）**：一个简单的奖励是——只要杆保持平衡且小车留在轨道上（例如 $|x_t| \le 2.4$ 且 $|\theta_t| \le 12^\circ$），每一步给 $r_t = 1$；任一边界被突破则回合终止。

- **动力学 / 转移（$p(s_{t+1}\mid s_t,a_t)$）**：在许多环境中动力学是确定性的（即 $p$ 为点质量），可用步长 $\Delta t$ 的欧拉积分写成 $s_{t+1} = f(s_t,a_t)$。一个标准的简化 CartPole 更新用到这些常量：小车质量 $m_c$、杆质量 $m_p$、杆半长 $l$、重力 $g$（$\alpha$ 是一个带加速度量纲、按质量归一化的中间量）：

  $$\alpha = \frac{a_t + m_p l\,\dot{\theta}_t^2\sin\theta_t}{m_c + m_p}$$ {#eq:cartpole_temp}

  $$\ddot{\theta}_t = \frac{g\sin\theta_t - \cos\theta_t\,\alpha}{l\left(\tfrac{4}{3} - \frac{m_p\cos^2\theta_t}{m_c + m_p}\right)}$$ {#eq:cartpole_angular_accel}

  $$\ddot{x}_t = \alpha - \frac{m_p l\,\ddot{\theta}_t\cos\theta_t}{m_c + m_p}$$ {#eq:cartpole_linear_accel}

  $$x_{t+1}=x_t+\Delta t\,\dot{x}_t,\quad \dot{x}_{t+1}=\dot{x}_t+\Delta t\,\ddot{x}_t,$$ {#eq:cartpole_pos_update}
  $$\theta_{t+1}=\theta_t+\Delta t\,\dot{\theta}_t,\quad \dot{\theta}_{t+1}=\dot{\theta}_t+\Delta t\,\ddot{\theta}_t.$$ {#eq:cartpole_angle_update}

这是前面通用设定（@eq:rl_opt）的一个具体实例：策略选择 $a_t$，转移函数推进状态，奖励在回合中累加。

### 改造标准 RL 设定

RLHF 的 RL 形式化被视为一个开放性更低的问题——为了适配语言模型，RL 中若干关键部件都被赋予了特定定义。
从标准 RL 设定到 RLHF 设定，有多处核心改动：
下表总结了标准 RL 与用于语言模型的 RLHF 设定之间的这些差异。

1. **从奖励函数转向奖励模型。** 在 RLHF 中，用一个从人类偏好学到的模型 $r_\theta(s_t, a_t)$（或任何其他分类模型）取代环境奖励函数。这大幅提升了方法的灵活性、也增强了设计者对最终结果的控制力，代价是实现复杂度上升。在标准 RL 中，奖励被视为环境的一个静态部件，设计学习算法的人无法改变或操纵它。
2. **不存在状态转移。** 在 RLHF 中，领域的初始状态是从训练数据集里采样的提示，而“动作”是对该提示的补全（在标准 RLHF 设定中，提示是固定的，模型的补全并不决定下一个提示）。一个提示加一个补全构成一个完整回合（rollout）——而在经典 RL 问题中，这会是许多重复的“状态-动作、状态-动作”链。
3. **回复级奖励，且不折扣。** RLHF 的奖励归因是针对整段动作序列（由多个生成的 token 组成）进行的，而不是细粒度地逐步进行（这种单步结构在 RL 文献中有时被称为赌博机问题）。为了让 RLHF 的 RL 算法把每个 token 都视为同一个动作的一部分，实现中通常取折扣因子 $\gamma = 1$（不折扣）——这与标准 RL 不同，后者用 $\gamma < 1$ 在许多连续决策之间平衡短期与长期奖励。

| 方面 | 标准 RL | RLHF（语言模型） |
|---|---|---|
| 策略 | 从零学习（随机初始化） | 从预训练语言模型微调而来 |
| 奖励信号 | 环境奖励函数 $r(s_t,a_t)$ | 学到的奖励 / 偏好模型 $r_\theta(x,y)$（提示 $x$，补全 $y$） |
| 状态转移 | 有：动力学 $p(s_{t+1}\mid s_t,a_t)$ | 通常没有：提示 $x$ 从数据集采样；补全不决定下一个提示 |
| 动作 | 单个环境动作 $a_t$ | 一个补全 $y$（一段 token 序列），从 $\pi_\theta(\cdot\mid x)$ 采样 |
| 奖励粒度 | 通常逐步 / 细粒度 | 通常是对整个补全的回复级（赌博机式），且通常不折扣（$\gamma = 1$） |
| 时间跨度 | 多步回合（$T>1$） | 常为单步（$T=1$），不过多轮对话可建模为更长的时间跨度 |
Table: 标准 RL 与语言模型 RLHF 之间的关键差异。 {#tbl:rl-vs-rlhf}

考虑到问题的单轮性质，可以把优化重写为不含时间跨度和折扣因子（并且带一个显式的奖励模型）：
$$\max_\pi \; \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t) \right].$$ {#eq:rl_opt_int}

其结果是：尽管 RLHF 深受 RL 优化器与问题形式化的启发，实际实现与传统 RL 却非常不同。

![标准 RLHF 循环](images/rlhf.png){#fig:rlhf data-dark-src=“images/rlhf-dark.png”}

### 微调与正则化

在传统 RL 问题中，智能体必须从一个随机初始化的策略开始学习；而 RLHF 是从一个具备许多初始能力的强大预训练基座模型出发。
这一强先验给 RLHF 带来了一个需求：必须防止优化偏离初始策略太远。
为了在微调体制下取得成功，RLHF 技术会采用多种正则化来控制优化。
目标是在避免模型陷入过度优化的前提下，仍然让奖励最大化发生（见第 14 章）。
对优化函数最常见的改动，是在当前 RLHF 策略与优化起点之间加一个 KL 散度惩罚。训练时设定的超参 $\beta$ 控制这一约束的强度——$\beta$ 越大，模型被约束得越靠近起点；$\beta$ 越小，优化器追逐奖励的自由度越大：

$$\max_\pi \; \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t)\right] - \beta  \mathcal{D}_{\text{KL}}(\pi(\cdot|s_t) \| \pi_{\text{ref}}(\cdot|s_t)).$$ {#eq:rlhf_opt_eq}

在这个框架下，RLHF 训练中相当多的研究都花在“如何花掉一笔固定的 KL 预算”上——这里所谓预算，是以偏离初始模型的距离来度量的。
更多细节见第 15 章「正则化」。


### 优化工具

本书会详述求解这一优化问题的许多流行技术。
后训练的常用工具包括：

- **奖励建模**（第 5 章）：训练一个模型来捕捉从偏好数据中收集到的信号，之后它就能输出一个标量奖励，指示未来文本的质量。
- **指令微调**（第 4 章）：RLHF 的前置步骤。通过模仿预先挑选的示例，教模型掌握当今大多数语言模型交互所用的问答格式。
- **拒绝采样**（第 9 章）：最基础的 RLHF 技术——用模仿人类偏好的奖励模型，筛选用于指令微调的候选补全。
- **策略梯度**（第 6 章）：RLHF 奠基性示例中使用的强化学习算法——依据奖励模型给出的信号，更新语言模型的参数。
- **直接对齐算法**（第 8 章）：直接用成对偏好数据优化策略的算法，而不是先学一个中间的奖励模型、之后再优化。

现代用 RLHF 训练的模型总是会用到指令微调，随后再混用其他优化选项。

### 在后训练语言模型中使用 RL 的隐性优势

接下来的章节会介绍许多后训练优化工具。
其中不少——例如拒绝采样（第 9 章）和 DPO 这类直接对齐算法（第 8 章）——都比“把 RL 跑起来”简单得多。
然而，尽管替代方案更简单，基于 RL 的方法依然胜出。
有些趋势是显而易见的，比如可验证奖励的强化学习（RLVR）带来的推理时扩展；但事实证明，RL 对语言模型来说是一种非常合适的优化工具。
实现 RL 所需的基础设施投入远高于指令微调或 DPO 类算法；但说得口语一点，它带来的梯度更新“总体上对模型帮助很大”。
这很难量化，但会以几种反复出现的形式体现出来：

> **原文注**：以下两条要点中，“numerical stability with inference tools like vLLM” 指的是让模型在与 vLLM 这类推理工具配合时保持数值稳定。

- RL 阶段可以“磨平”模型的粗糙边角，让它更易于对话、更鲁棒（例如通过训练，使其在与 vLLM 这类推理工具配合时数值更稳定）。文献中对确切原因尚无定论，但今天 RL 越来越普遍的采用，本身就印证了这一点。
- RL 可以“精准施治”——模型很擅长学会提示分布落在哪里，而 RL 往往不会“压坏”模型的通用能力。一个很好的例子是 Tülu 3：只在数学提示上做 RL 训练，却能在广泛的任务套件上维持原有能力 [@lambert2024t]。

总体而言，语言模型上的 RL 损失是鲁棒、可扩展、有效且灵活的，这打开了若干新的大规模实验领域。
开启这条路的原始方法，正是 RLHF 的工作。

## 经典训练流程

随着时间推移，若干模型被公认为 RLHF（或更广义的后训练）的经典流程范例。
这些流程反映了当时的数据实践与模型能力。
随着流程老去，训练出同等特性的模型会变得更容易，所需数据也更少。
总的趋势是：后训练涉及的优化步骤越来越多、训练算法越来越多、训练数据与评估也越来越多样。

### InstructGPT：奠基性的 RLHF 工具

在 ChatGPT 刚问世的那个时期，被广泛接受的（“经典的”）语言模型后训练方法有三个主要步骤，其中 RLHF 是核心 [@lambert2022illustrating] [@ouyang2022training] [@bai2022training]。
在一个“基座”语言模型（在大规模网页文本上训练的下一词预测模型）之上进行的这三个步骤，汇总于 @fig:rlhf-basic-repeat：

1. **在约 1 万条样本上做指令微调**：教模型遵循问答格式，并从主要由人撰写的数据中习得一些基础技能。
2. **在约 10 万个提示（含成对补全）上训练奖励模型**（论文实际用了 3.3 万个提示）：这个模型从指令微调的检查点出发训练，捕捉你在最终训练中希望建模的多样价值取向。奖励模型就是 RLHF 的优化目标。
3. **在另一批约 10 万个提示上用 RLHF 训练指令微调模型**（论文精确使用了 3.1 万个，且未说明提示是否与其他阶段复用）：模型针对奖励模型进行优化，提示很可能来自另一批数据；模型先生成回复，再接收评分。

RLHF 完成后，模型就可以部署给用户了。这个流程是现代 RLHF 的基础，但流程已经大幅演化，包含更多阶段和更多数据。

![早期三阶段 RLHF 流程示意图：SFT、奖励模型，然后是优化。](images/rlhf-basic.png){#fig:rlhf-basic-repeat}

### Tülu 3：把 RLVR 引入指令模型

现代版本的后训练涉及多得多的模型版本与训练阶段（远多于 Llama 2 所记录的 5 个 RLHF 步骤 [@touvron2023llama]）。
一个例子见 @fig:rlhf-complex，其中模型在收敛前经历了大量训练迭代。

![现代多轮后训练的示意图。](images/rlhf-complex.png){#fig:rlhf-complex}

这个时代及以后训练的最复杂模型，并未公布其训练过程的完整细节。
到 2026 年，ChatGPT 或 Claude 这类领先模型都涉及许多轮迭代训练。
其中甚至可能包含这样的技术：先训练多个专门化模型，再把权重合并，得到一个能胜任多种子任务的最终模型 [@li2022branch]（例如 Cohere 的 Command A [@cohere2025command]）。

![Tülu 3 流程汇总，含目标技能与多步训练配方。Lambert 等 2024，许可 CC-BY。](images/tulu3.png){#fig:tulu-3}

这种多阶段后训练方法（RLHF 在其中扮演主要角色）的一个完全开放的范例是 Tülu 3。
Tülu 3 流程由三个阶段组成：

1. **在约 100 万条样本上做指令微调**：这个以合成数据为主的数据集取自 GPT-4o、Llama 3.1 405B 等前沿模型的混合输出，教模型通用的指令跟随能力，并为数学、编程等能力打下基础。
2. **在约 100 万对偏好数据上做同策略（on-policy）偏好优化**：这一阶段大幅提升模型的“会话感”（例如 Arena——前身为 Chatbot Arena——或 AlpacaEval 2 上的表现），同时也改善指令微调阶段提到的那些技能。
3. **在约 1 万个提示上做可验证奖励的强化学习**：这是一个小规模强化学习运行，用于提升数学等核心技能、同时维持整体性能（如今它被视为 DeepSeek R1 等现代推理模型的先声）。

该流程已成功应用于 Llama 3.1 [@lambert2024t]、OLMo 2 [@olmo20242] 和 SmolLM 系列模型 [@alrashed2024smoltulu]。

### DeepSeek R1：为推理而扩展 RLVR

随着 OpenAI o1 等推理语言模型的兴起，后训练的最佳实践再次演化：重新安排并重新分配各训练阶段的算力。
关于推理模型后训练流程，目前最清晰的公开记录是 DeepSeek R1 [@guo2025deepseek]；阿里巴巴规模更大的 Qwen 3 模型（仅指 32B 与 225B MoE 版本）[@yang2025qwen3]、以及小米的 MiMo 7B [@xia2025mimo] 也沿用了类似做法。

![DeepSeek-R1 的多阶段流水线。出自《Nature》上的 DeepSeek R1 论文，[图 2](https://www.nature.com/articles/s41586-025-09422-z/figures/2)，依 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) [@guo2025deepseek]。](images/deepseek-r1-pipeline.png){#fig:deepseek-r1-pipeline .center}

DeepSeek 的流程（见 @fig:deepseek-r1-pipeline）如下：

1. **用 10 万余条同策略推理样本“冷启动”**：这些数据采样自更早的 RL 检查点 R1-Zero，并经过大量筛选，以便在 DeepSeek-V3-Base 上注入特定的推理过程。DeepSeek 用“冷启动”一词来描述在少量监督数据下开始 RL 学习的方式。
2. **大规模强化学习训练**：这一阶段针对各种基准反复让模型接触推理问题，运行 RLVR“直到收敛”。
3. **拒绝采样与 SFT**：接近收敛时，他们对 RL 检查点做拒绝采样，构建约 80 万条样本的 SFT 数据集；然后在约 3/4 推理问题、1/4 通用查询的筛选混合数据上微调模型，得到一个通用模型。
4. **在推理问题（可验证奖励）上做混合强化学习训练**，并配合通用偏好微调的奖励模型来打磨模型。

如上所述，该流程也有多种演化，尤其体现在第 3、4 步如何最终定型模型、再暴露给用户。
许多模型会从定制化的指令数据集（包含思维链序列、由现有模型大量筛选与打磨）入手，从而仅凭 SFT 就能快速达到较强行为，然后再进入 RL [@seed2025seed]。

### 转向 MOPD 与智能体

推理模型一个关键的产品市场契合点，是向编程智能体的演化——这带来了经典后训练流程的又一次变化。
有两处关键变化。
第一，另一种新训练方法——多教师同策略蒸馏（MOPD，见第 12 章）——成为把多种不同技能合并进最终模型的流行工具。
第二处更简单：后训练算力持续快速扩展，尤其是在强化学习阶段。

小米的 MiMo-V2-Flash 是首个记录现代 MOPD 过程的技术报告 [@mimo2025flash]，团队后来还就此写了独立论文 [@ma2026mopd]。该技术报告用一种简洁的形式概括了这类后训练流程，见 @fig:rlhf-mopd：先做通用 SFT，再做领域专精的教师训练（更多 SFT 与大规模 RL），然后用多教师同策略蒸馏（MOPD）把它们合并成最终模型。
其他模型也采用这一形式的流程，例如 NVIDIA 的 Nemotron 3 Ultra [@nvidia2026nemotron3ultra]——大体相似，只是连续做了两轮跨专家的 MOPD 阶段。
总体而言，这类流程在“迭代模型版本”这一维度（配方的*深度*）上复杂度更低，但在“需要专精出许多关键专家”这一点（一种*广度*）上复杂度更高。需要注意的是，前几代后训练同样处理了广泛的任务，只是在训练过程中切分得没有那么细——也就是说，大多数任务是在相同的训练阶段里完成的。

![使用 MOPD 的专精式后训练示意图：共享 SFT、领域专属 SFT 与 RL，然后经多教师同策略蒸馏合并为一个（最终的）学生模型。](images/rlhf-mopd.png){#fig:rlhf-mopd data-dark-src=“images/rlhf-mopd-dark.png”}

以 MOPD 作为最后阶段的各种模型，其专家数量各不相同：Nemotron 3 Ultra 超过十个 [@nvidia2026nemotron3ultra]，Kimi K3 为九个 [@kimiteam2026kimik3]，DeepSeek V4 超过十个 [@deepseekai2026deepseekv4]（小米 MiMo V2 未报告专家数量）。

后训练流程在不同模型之间仍然差异很大。
MOPD 是一个流行的新工具，但各实验室对它的采用远未达成一致。随着后训练走向成熟、工具箱里的工具越来越多，流程的差异反而变大了。
例如 GLM-5 使用了 MOPD，但在 MOPD 之前记录了一个更复杂的三阶段 RL 过程，见 @fig:rlhf-sequential-rl [@glm5team2026glm5]。

![GLM-5 的整体训练流水线。出自 GLM-5 团队论文，[图 5](https://arxiv.org/pdf/2602.15763v2#page=4)，依 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) [@glm5team2026glm5]。](images/glm5-pipeline.png){#fig:rlhf-sequential-rl}

随着后训练在现代模型性能中占据更核心的位置，这些流程将继续演化，以反映难度、任务与目标所达到的新规模。
