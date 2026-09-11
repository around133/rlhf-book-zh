<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
  代码块与英文提示词保留原文。
-->
---
prev-chapter: "偏好数据"
prev-url: "11-preference-data"
page-title: 合成数据与蒸馏
search-title: "第 12 章：合成数据与蒸馏"
meta-description: "现代后训练中使用的合成数据、蒸馏、宪法式 AI 与 AI 反馈方法。"
next-chapter: "工具使用与函数调用"
next-url: "13-tools"
lectures:
  - video: "https://www.youtube.com/watch?v=6nyJ8y8ghsE&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=10"
    label: "第 7 讲：合成数据与现代后训练方法"
---

# 合成数据与蒸馏

基于*人类反馈*的强化学习，深深植根于“让我们所造的模型保持人类影响”这一理念。
当第一批模型成功用 RLHF 训练出来时，人类数据是**唯一**能用这种方式改进模型的办法。

人类是当时唯一能造出足够高质量回答用于训练的方式。
人类是当时唯一能采集可靠、具体的反馈数据来训练奖励模型的方式。

随着 AI 模型变强，这一假设迅速瓦解。
合成数据的可能性——更便宜、更容易迭代——通过降低实验与研究的成本，推动了 RLHF 的普及。
这也让 RLHF 成为更广义的“后训练”塑造模型思路中最早的中心。
本章粗略概览合成数据如何、以及为何正在取代或扩展 RLHF 流程的许多环节。

## 合成数据的角色

对合成数据的一个常见批评是**模型崩塌**（model collapse）——认为反复在模型自己的生成上训练，会逐步收窄有效的训练分布 [@shumailov2024ai]。
随着多样性下降，罕见事实与风格被低估，小的错误会在迭代中被放大，导致泛化能力变差。
实践中，这些失效最常与“在未过滤、重复、单模型输出上做自训练”相关联；而混入真实/人类数据、使用多样化的教师模型、去重、以及强力的质量过滤，基本可以避开崩塌区间。
对于今天的前沿训练流水线，证据表明合成数据能够、也应该被大规模使用，而不会出现“崩塌论最强版本”所暗示的灾难性退化 [@gerstgrasser2024model] [@feng2024beyond]。

领先模型**需要合成数据**才能达到最佳性能。
现代后训练中的合成数据涵盖训练的许多环节——语言模型被用来从种子示例生成新训练提示 [@wang2022self]、改写已有提示、生成对提示的补全 [@numina_math_7b]、提供 AI 反馈以构造偏好数据 [@cui2023ultrafeedback]、筛选补全 [@li2024superfiltering] 等等。
合成数据是后训练的关键。

合成数据能达到这种程度的影响力，是随着 GPT-4 级模型才出现的。
在早期语言模型（如 Llama 2 与 GPT-3.5-Turbo）上，模型在生成或监督数据流水线方面还不够可靠。
而在一两年之内，语言模型在生成回答上就已远超人类。
在从 GPT-3.5 向 GPT-4 级模型过渡的过程中，模型执行“LLM 作为评判者”任务的能力也出现了。
GPT-4 或更强的模型在针对某段内容生成反馈或评分时，要稳健、一致得多。

自 2022 年末 ChatGPT 发布以来的这些年里，我们见证了许多有影响力的合成数据集。
包括 UltraFeedback [@cui2023ultrafeedback]——第一个著名的合成偏好数据集，它点燃了 DPO 革命；2023 年的 Stanford Alpaca，最早的聊天式微调数据集之一；Tülu 3 中面向技能（如数学、代码、指令跟随）的合成数据集 [@lambert2024t]；以及 2025 年为训练思考模型而生的 OpenThoughts 3 和许多其他合成推理数据集 [@guha2025openthoughts]。
如今要上手工业级后训练，多数经典参考都涉及上面这类 Tülu 3 或 OpenThoughts 3 数据集；但快速上手指南常常从更小更简单的数据集（如 Alpaca）开始，因为训练快得多。

一个大的变化与数据集规模有关：微调数据集的提示数量在增长——Alpaca 是 5.2 万条，OpenThoughts 与 Tülu 3 是 100 万条以上——回复长度也在增长。
更长的回复加上更多提示，使 Alpaca 数据集大约在 1000 万训练 token 量级，而 Tülu 是它的 50 倍、约 5 亿 token，OpenThoughts 3 更大，约在 100 亿 token 量级。

在这一整个转变过程中，合成数据并没有在流水线各处均匀地取代人类数据。
对**指令数据（SFT）**，合成生成基本胜出——从更强模型蒸馏出的补全，在规模上比大多数人类写手能提供得更好（在最困难的前沿推理问题上有些例外）。
对 **RLHF 中的偏好数据**，情况更混合：学术工作表明合成偏好数据表现相当，但前沿实验室仍把人类偏好数据视为竞争壁垒。
对**评估**，分化呈现另一种形态：LLM 作为评判者能以低成本扩展对模型输出的*打分*，但底层基准与真值标签仍需人类创建。
规律是：**在模型可靠性超过人类的环节，合成数据占主导；而在能力前沿、确立真值、以及引导训练方面，人类仍不可替代。**

## 用合成数据做蒸馏

“蒸馏”这个词，是关于合成数据在语言模型中作用的最有力的一类讨论。
蒸馏作为一个术语，来自深度学习文献中教师—学生**知识蒸馏**（Knowledge Distillation，KD）的技术定义 [@hinton2015distilling]。

![传统知识蒸馏训练一个更小的学生模型，用 KL 散度损失去匹配更大教师模型的软概率分布。两个模型同时处理同一输入，温度缩放（$\tau > 1$）柔化分布以揭示更多关于类别关系的信息。](images/knowledge_distillation_tikz.png){#fig:knowledge-distillation data-dark-src=“images/knowledge_distillation_tikz-dark.png”}

口语上的蒸馏，指的是用更强模型的输出去训练一个更小的模型。

![LLM 后训练中的合成数据生成：提示经过一个强模型生成补全，再配对构成训练数据集。该数据集随后通过标准监督学习用于微调更小的模型。更复杂的流水线可能涉及多个模型编辑补全、生成偏好对，或做质量过滤。](images/synthetic_data_distillation_tikz.png){#fig:synthetic-data-generation data-dark-src=“images/synthetic_data_distillation_tikz-dark.png”}
在后训练中，这种广义的蒸馏有两种常见形式：

1. 作为**数据引擎**，横跨后训练流程的广阔环节：指令的补全、偏好数据（或宪法式 AI），或用于 RL 的验证。
2. 把特定技能从强模型迁移到弱模型，常常针对数学推理或编程这类具体技能。

第一种策略随着语言模型在“为各种任务撰写答案”上变得比人更可靠而日益流行。
GPT-4 级模型把范围扩展到了用强模型蒸馏来处理数学、代码这类复杂任务（如上所述）。
在这里，蒸馏促成了“模型套件”的形态：实验室往往训练一个大型内部模型——如 Claude Opus 或 Gemini Ultra——不公开释放，仅供内部用来造更强的模型。
对开放模型，常见做法是把闭源 API 模型的训练数据蒸馏到更小、可公开获取的权重里 [@tunstall2023zephyr]。
其中，策展高质量提示、筛选教师模型的回答，对最大化性能至关重要。

把特定技能迁移到更小的语言模型，用的是同样的蒸馏原则——拿到尽可能好的训练数据。
这方面，许多论文研究了用来自强模型的有限数据集来改进对齐 [@zhou2023lima]、数学推理 [@shridhar2023distilling] [@hsieh2023distilling]，以及测试时扩展 [@muennighoff2025s1]。

本章余下的合成数据方法，都是“构造把语言模型输出直接用在训练流水线里的数据配方”的方式。

## 通往同策略教师—学生蒸馏之路

尽管蒸馏总体上已成为后训练语言模型的标准做法，随着后训练配方转向推理与智能体模型，**教师—学生知识蒸馏**这一具体子领域又重新受到关注。
用新形式知识蒸馏训练的领先模型例子包括阿里巴巴的 Qwen3 [@yang2025qwen3]、小米的 MiMo-V2-Flash [@mimo2025flash]、智谱 AI 的 GLM-5 [@glm5team2026glm5]，以及 DeepSeek-V4-Pro [@deepseekai2026deepseekv4]。

蒸馏之所以属于本章，是因为现代后训练中合成数据的许多用法，实践中就是**受蒸馏启发的流水线**：更强的模型产出标签、补全、logits、评语或其他监督，学生模型在这些信号上训练。
与此同时，关于蒸馏的技术文献正在成长为它自己的一套后训练方法，尤其随着同策略与自蒸馏配方越来越常见。
目前我们把它作为合成数据工具箱的一部分在此覆盖；但本书未来的版本，可能值得为“蒸馏作为一种训练工具”单开一章，与指令微调、强化学习等并列。

### 为语言模型改造知识蒸馏

最初的文献提出知识蒸馏，专门是为了从已训练好的、更强和/或更大的**教师**网络训练一个**学生**模型 [@hinton2015distilling]。
KD 以使用*软*训练标签著称，而不是像“下一词预测 + 交叉熵损失”这类标准目标中使用的独热标签。
软标签上的目标关注所有可能下一 token（或预测）上的分布，而不只是那一个被预测的 token 对不对；它训练学生分布去匹配教师分布。

KD 一般可应用于任何深度学习问题，例如预测输入所属的单个类别。
要专门应用到语言模型的自回归风格上，损失可以分解为逐 token 的分布匹配损失。
2016 年，Kim 与 Rush 把 KD 应用于让学生模型从教师模型生成的*序列*中学习 [@kim-rush-2016-sequence]。

令 $s$ 为源句或提示，$u = (u_1,\ldots,u_J)$ 为教师模型产生的完整输出序列，$\mathcal{V}$ 为输出词表（tokenizer 中可能的 token），$q$ 为教师在下一 token 上的分布，$p$ 为学生分布。
这里我们用 $u$ 作为“教师完整输出序列”的中性符号，把 $a$ 留给下文同策略/RL 记法中“学生采样的补全/动作序列”。
注意他们的论文称之为“词级蒸馏”，但对现代语言模型，最好把它读作**在 tokenizer 词表上的逐 token 分布匹配**，因为该论文早于现代子词 tokenizer：

$$
\mathcal{L}_{\mathrm{WORD-KD}}
= -\sum_{j=1}^{J}\sum_{k=1}^{|\mathcal{V}|}
q(u_j = k \mid s, u_{<j})\log p(u_j = k \mid s, u_{<j}).
$$ {#eq:word_kd}

WORD-KD 是把经典的、受 Hinton 启发的教师—学生知识蒸馏应用到语言模型上。这通常是在一段已经在训练语料中的静态文本上进行的。

它具有普通的交叉熵形式 $-\sum_z q(z)\log p(z)$。
在每个位置 $j$，教师分布 $q$ 给每个可能的下一 token $k \in \mathcal{V}$ 分配概率，而每当学生的分布 $p$ 给教师认为可能的 token 分配低概率时，学生就受到惩罚。

序列级蒸馏则把 $\mathcal{U}$ 当作可能输出序列的空间，让学生去匹配教师在**完整序列**上的分布。
由于对所有完整序列 $u \in \mathcal{U}$ 求和不可行（需要对指数级数量的潜在序列求和），Kim 与 Rush 用一个“点质量”近似教师在序列上的分布——即在单个高概率的教师输出 $\hat{u}$ 上取点质量。
这里 $\hat{u}$ 是用教师模型做束搜索得到的序列，所以 $\hat{u} = \mathrm{BeamSearch}_q(s) \approx \arg\max_{u \in \mathcal{U}} q(u \mid s)$：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{SEQ-KD}}(s)
= -\sum_{u \in \mathcal{U}} q(u \mid s)\log p(u \mid s)
\approx -\log p(\hat{u} \mid s) \\
= -\sum_{j=1}^{|\hat{u}|}\log p(\hat{u}_j \mid s, \hat{u}_{<j}).
\end{aligned}
$$ {#eq:sequence_kd}

SEQ-KD 朝现代方法迈出了一步：教师模型在**生成 token** 作为学生的信号。这是解锁后文同策略蒸馏各种形态的核心一步，也是让“对所有可能序列求和”变得可行的必要手段。
当我们转向现代模型上流行的 KD 变体时，我们把这种训练风格称为*离线* KD——即用于训练学生模型的生成是**事先**产生的。

在继续之前，有两个联系值得指出。

第一，有一系列用离线 KD 训练的流行模型，例如分类模型 DistilBERT [@sanh2019distilbert] 与 TinyBERT [@jiao2020tinybert]，它们把语言模型的其他改进与离线蒸馏结合起来（注意，不是*序列*蒸馏，因为这些编码器模型并不为多 token 自回归预测而蒸馏）。

第二，我们可以与第 15 章对 KL 散度的详尽讨论建立联系，因为上面用的交叉熵目标与 KL 散度密切相关。
对教师分布 $q$ 与学生分布 $p$，交叉熵定义为

$$
H(q,p) = -\sum_z q(z)\log p(z).
$$ {#eq:kd_cross_entropy}

它的形式与 @eq:word_kd 相同，也与 @eq:sequence_kd 的第一项相同。
交叉熵还可以分解为教师分布的熵加上一个 KL 散度：

$$
\begin{aligned}
H(q,p)
&= H(q) + D_{\mathrm{KL}}(q\|p) \\
&= -\sum_z q(z)\log q(z)
+ \sum_z q(z)\log\frac{q(z)}{p(z)}.
\end{aligned}
$$ {#eq:kd_forward_kl}

第一项 $H(q)$ 只依赖教师。
因此，当教师固定、并且它就是训练数据的来源时，最小化交叉熵等价于最小化从教师到学生的**前向 KL** $D_{\mathrm{KL}}(q\|p)$。
这正是离线 KD 与类 SFT 训练所用的 KL 方向。

### 从离线蒸馏到同策略蒸馏

这些*离线* KD 算法有一些局限，促成了同策略变体的出现。
学习的离线性质意味着，学生模型可能面临“教师模型”与“学生推理时自己生成的序列”之间的分布不匹配。
例如，前向 KL 目标会把学生模型推向高估教师分布中的低概率区域。
这些问题合在一起，为*同策略*蒸馏（OPD）开了口子。

这种“训练—测试差距”被称为**暴露偏差**（exposure bias）[@arora-etal-2022-exposure] [@song2026surveyonpolicydistillationlarge]。
离线 KD 采样教师的轨迹 $u \sim \pi_T(\cdot \mid s)$，并在由此得到的前缀上最小化逐 token KL，

$$
\mathcal{L}_{\mathrm{KD}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D},\, u \sim \pi_T(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\!\left(
\pi_T(\cdot \mid s, u_{<t})
\;\|\;
\pi_\theta(\cdot \mid s, u_{<t})
\right).
$$ {#eq:exposure_train}

而在推理时，学生是在**自己的**策略下 rollout，所以真正重要的是沿着*它自己*轨迹的期望任务损失，

$$
\mathcal{L}_{\mathrm{eval}}(\theta)
= \mathbb{E}_{s \sim \mathcal{D}_{\mathrm{test}},\, a \sim \pi_\theta(\cdot \mid s)}
\ell_{\mathrm{task}}(s, a)
$$ {#eq:exposure_test}

这里 $\ell_{\mathrm{task}}(s, a)$ 表示对学生完成的回复的任何下游任务损失，例如答案错误、测试用例失败，或评判/评分准则损失。
暴露偏差是 $\pi_T(\cdot \mid s) \neq \pi_\theta(\cdot \mid s)$ 这一不等式的直接后果：训练时访问的前缀 $(s, u_{<t})$ 与测试时访问的前缀 $(s, a_{<t})$，来自不同的状态访问分布，所以学生是在一组**与它实际行动所在不同**的状态上被监督的。

转向同策略蒸馏的核心想法是：我们可以通过**从学生模型采样、并度量它与教师分布的距离**来改变优化，而不是从教师模型采样。
MiniLLM 指出了转向反向 KL 优化的必要性（我们会在第 15 章直观解释为什么这个目标可能更好），并提出把 KD 损失函数放进一个在线策略梯度 RL 框架中使用 [@gu2024minillm]。
其他同期工作 [@agarwal2024policy] 展示了同策略 KD 的前景，并把“从学生生成、由教师打分”这一迭代过程与 RL 文献中的模仿学习工作联系起来。
建立这个联系：其中一种模仿学习算法 DAgger，迭代地训练一个“用自己学到的策略在世界上行动、并从专家策略处获得'本应采取什么动作'的反馈”的智能体，再用这些反馈更新其策略 [@ross2011reduction]。

这一差距的代价，可以通过激励 DAgger 的监督式模仿学习界来量化。
在原始的离散动作设定下，假设学到的策略在教师诱导的训练分布上的期望逐步动作误差不超过 $\epsilon$，其中 $\mathbb{I}[\cdot]$ 是指示函数（条件为真时返回 1，否则 0），

$$
\mathbb{E}_{s_t \sim d_{\pi_T}}\!\left[
\mathbb{I}\!\left(\pi_\theta(s_t) \neq \pi_T(s_t)\right)
\right] \leq \epsilon.
$$ {#eq:dagger_perstep}

监督式模仿学习的分析 [@ross2011reduction] 表明：沿着从学生采样的、长度为 $L$ 的轨迹累积的期望损失，可以随 $L$ **二次**增长 [@song2026surveyonpolicydistillationlarge]：

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\!\left[\sum_{t=1}^{L} \ell\!\left(s, a_{<t}\right)\right] \leq O(\epsilon L^2).
$$ {#eq:dagger_trajectory}

对 LLM 而言，这个离散动作的界应当读作**类比**，而不是理论保证。
实践中，LLM 在很长的跨度上预测完整的下一 token 分布，所以 @eq:dagger_perstep 中“动作是否一致”的 0-1 假设并不严格适用。
提示或前缀自然地对应状态，采样的 token 对应动作，但逐 token 蒸馏通常用 KL 或交叉熵这类**分布型**损失来度量，所以经典的 DAgger 数学并不能精确迁移。

这种 $O(\epsilon L^2)$ 式的误差复合，对现代 LLM 尤其明显——它们例行生成跨越数千 token 的序列。
单个次优 token 会把前缀略微推离分布，而模型从未见过这个被扰动的前缀，于是更可能再次出错，最终产生退化或幻觉文本。
同策略蒸馏通过*迭代地*从当前学生采样补全、并由教师在这些被访问到的状态上施以监督，来解决这个问题。
学生直面自己的错误，在自己访问到的具体分布外状态上获得教师反馈，并学会恢复行为。
在 DAgger 的交互式模仿学习分析下，这一迭代过程可以把误差复合从 $O(\epsilon L^2)$ 降低到 $O(\epsilon L)$ [@ross2011reduction]。
对 LLM 而言，这解释了 OPD 背后的动机：精确的界未必能干净地迁移到每一种逐 token 蒸馏设置上，但同策略方法的实践成功支持了底层的直觉。

对同策略蒸馏，令 $s$ 为提示，$a = (a_1,\ldots,a_L)$ 为从当前学生策略 $\pi_\theta(\cdot \mid s)$ 采样的补全，$s_t = (s, a_{<t})$ 为第 $t$ 步的 token 级状态。
教师策略 $\pi_T$ 固定，因此目标是在**由学生诱导的状态**上，把学生的下一 token 分布与教师的分布作比较。
由于期望是从 $\pi_\theta$ 采样、且学生分布位于 $D_{\mathrm{KL}}(\pi_\theta \| \pi_T)$ 的左侧，这是一个反向 KL 目标：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
= \mathbb{E}_{s, a \sim \pi_\theta(\cdot \mid s)}
\sum_t D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid s_t) \;\|\; \pi_T(\cdot \mid s_t)\right).
$$ {#eq:opd_reverse_kl}

这里我们转向期望记号——正如第 6 章（讲述基础 RL 策略梯度算法的那一章）大量使用的那样——因为该优化是通过采样轨迹并数值估计梯度来求解的。
这一转向采样框架，也自然地过渡到现代 LLM 的 RL 训练基础设施：它的设计就是为了在“从当前被训练策略生成 token”与“做学习更新”之间快速交替。

事实上，近期 OPD 的实现把 KD 与 RL 的这种整合又推进了一步：把 KD 距离直接当作 RL 优化中的奖励信号。
一个经典做法，是把“反向 KL 的逐 token 贡献取负”作为某 RL 算法中的优势 [@lu2025onpolicy]。
对状态 $s_t$ 处采样得到的 token $a_t$，逐 token 的对数概率差可以写成一个类优势信号：

$$
A_t^{\mathrm{OPD}}
= \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t).
$$ {#eq:opd_kl_advantage}

用逐 token KL 贡献的负值，就把最小化转成了最大化信号：教师评价高于学生的被采样 token 获得正优势，教师评价低于学生的 token 获得负优势。
教师的对数概率差就像**稠密的 token 级反馈**，可能比稀疏的可验证奖励或奖励模型输出提供更有用的学习反馈。

### 现代 OPD 变体

这个设定还能进一步扩展：用多个教师模型共同教一个最终模型，或在生成中插入额外信息以帮助模型识别错误。
首先，我们介绍如何把多个教师整合进一次训练运行。
这些教师可以是特定领域的专家模型（例如数学或代码领域），也可以是先前的某个中间训练检查点。
对每个教师，可以按提示或批次中的任务类型选择贡献权重，从而构造**多教师同策略蒸馏**（MOPD）[@mimo2025flash]。
对多个教师，令 $\pi_{T_k}$ 为教师 $k$、$w_k(s)$ 为其依赖提示的混合权重（且 $\sum_k w_k(s) = 1$），反向 KL 损失为：

$$
\mathcal{L}_{\mathrm{MOPD}}(\theta)
= \mathbb{E}_{s, a \sim \pi_\theta(\cdot \mid s)}
\sum_t \sum_k w_k(s) D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid s_t) \;\|\; \pi_{T_k}(\cdot \mid s_t)\right).
$$ {#eq:mopd_objective}

在大规模后训练中，这能让不断扩张的组织进一步扩展其配方。
多个团队可以各自打磨高质量的专家模型，这些模型日后可作为最终学生模型的教师——[@deepseekai2026deepseekv4] 与 [@mimo2025flash] 就是这么做的。

把 OPD 与本书研究的其他方向结合起来的方式有很多，例如把反向 KL 作为优势之一，与其他形式的优势计算（如 GRPO 的组级归一化）并用，从而实现更复杂的奖励整形。
KD 方法在后训练方法中比较特殊，因为它通常要求学生与教师**共享同一个 tokenizer**——毕竟监督可以是来自另一个 LLM 的逐 token 反馈。

更进一步的方案，例如**同策略自蒸馏**（On-Policy Self-Distillation，OPSD），让一个语言模型自己（或借助外部工具）去验证某个补全，从而充当一个拥有**特权信息**的教师，这样它就能在没有一个明确更强教师的情况下提升自己 [@zhao2026selfdistilled]（OPSD 训练的概览见 @fig:sdpo）。
例如，Cursor 用自蒸馏的形式——对 RL 轨迹给出针对性的文本反馈——来训练它的 Composer 2.5 编程模型 [@cursor2026composer25]，后者从 Kimi K2.5 微调而来。
下面是一个简化的直觉说明；实践中，下面的设定会与其他损失函数（例如代码正确性）结合使用。
在这个设定里，Cursor 让模型用一个包含常见 bug 清单的评判提示去复查 RL 轨迹。
遇到 bug 时，评判模型会修改 RL 中已生成的序列——插入一条供模型将来学习的提示——然后继续做蒸馏损失。
这构成一个循环：先用标准语言模型生成在 RL 中产出一个补全，然后跑评判模型并可选地插入一个提示 token，最后对新补全生成 logprobs 以施加知识蒸馏损失。
对模型而言，token 空间里的这条提示就足以帮助它纠正自己的输出——即便是在性能的绝对前沿上改进时也是如此（关于如何最好地构造与使用这些提示，仍有大量持续工作，它们常被称为*特权信息* [@penaloza2026privileged]）。

这使同策略蒸馏成为核心的后训练方法：它可用于把多种技能合并进一个通用模型，或在专门化部署中推进性能前沿。

![三种蒸馏范式，按 rollout 从哪来、监督如何流动来对比。**序列 KD**（左）：教师离线生成输出，学生用交叉熵（CE）损失去匹配它。**同策略蒸馏（OPD）**（中）：学生在同策略下生成 rollout（例如在 RL 框架内），另一个教师为每个被访问的 token 打分，用逐 token 的 KL 散度（KL）训练学生。**同策略自蒸馏（OPSD）**（右）：一个模型同时扮演两个角色——把特权信息（提示）加入上下文形成教师轨迹，再用 KL 损失把无提示的生成蒸馏向它，全程没有独立的教师模型。](images/distillation_directionality_tikz.png){#fig:distillation-directionality data-dark-src=“images/distillation_directionality_tikz-dark.png”}

![字符串反转任务上的同策略自蒸馏（OPSD）。同一个策略 $\pi_\theta$ 对同一条学生采样的补全 $y$ 做两次前向：一次**教师**前向，条件里带问题加一个正确的同类示例（黄色）；一次**学生**前向，只带问题（绿色）。两次前向之间的逐 token 反向 KL，对教师一侧取 stop-gradient，把“只带问题”的策略拉向“带示例”的自己；高亮列是分布差异最大、采样错误的那些 token。](images/sdpo_tikz.png){#fig:sdpo data-dark-src=“images/sdpo_tikz-dark.png”}

### 建议的实验

`code/distillation/` 中的配套代码实现了 SDPO [@hubotter2026reinforcement]，也就是 @fig:sdpo 展示的同策略自蒸馏设定（同期的 OPSD 论文 [@zhao2026selfdistilled] 与之密切相关）：一个策略同时充当“带示例的教师”和“只带问题的学生”，用逐 token 反向 KL 训练。
它跑在一个小型的字符串反转任务上，这让同策略蒸馏循环便宜到可以在单张 GPU 上端到端观察。

1. **跑一遍 SDPO 字符串反转示例。**

   ```bash
   cd code/
   uv run python -m distillation.train --config distillation/configs/sdpo.yaml
   ```

   观察 `reward`、`loss` 和 `skipped`，以及循环里打印的教师/学生 rollout 样本。
   `skipped` 计数是“被轮询提示中、其采样组里没有任何一条正确 rollout”的数量；随着学生变好，被跳过的提示减少，`reward` 朝 1 攀升。

2. **调整同策略相关的旋钮。**
   复制 `distillation/configs/sdpo.yaml`，在任务固定的前提下扫 `num_rollouts`、`kl_top_k` 和 `prompts_per_step`。
   每个提示更多 rollout 更容易找到一个正确的同类示例（降低 `skipped`），代价是每步生成更多；`kl_top_k` 则在“反向 KL 匹配教师分布的比例”与算力之间做权衡。

## AI 反馈

在 RLHF 快速增长之后不久，**AI 反馈强化学习**（RLAIF）作为一种替代路径出现：让 AI 来近似流水线中“人类数据”那一环，从而加速实验或进展。
广义上，AI 反馈是一大类技术，用 AI 来增补或生成“解释某个输入质量”的数据（可用于不同的训练方法或评估）；它始于成对偏好 [@lee2023rlaif] [@sharma2024critical] [@castricato2024suppressing]。
使用 RLAIF 来完全取代人类反馈、或增补它，动机有很多。
在 RLHF 流程中，AI 反馈最著名的作用是在偏好数据采集及相关的奖励模型训练阶段（宪法式 AI 就是其中一种具体实现）。
本章聚焦一般性的 AI 反馈，以及它在 RLHF 训练流水线中的这种具体用法；本书后文还会覆盖理解或使用合成数据的更多方式。

随着 AI 反馈走向成熟，它的应用范围扩展到了“仅仅替代人类偏好标签”之外。
同一套支撑了更廉价偏好数据采集的“LLM 作为评判者”基础设施，也支撑了可扩展的评估（见第 16 章）；而更近期，它还支撑了**基于评分准则（rubric）的奖励**——把 RL 训练扩展到没有可验证答案的领域，这一前沿会在本章后文探讨。

### 平衡 AI 反馈与人类反馈数据

在生成给定数量的反馈上，AI 模型远比人类便宜：截至 2026 年，单条人类偏好数据的成本在 1 美元量级或更高（有些甚至每个提示超过 10 美元），而用 GPT-4o 这类前沿 AI 模型做 AI 反馈，成本不到 0.01 美元。
除此之外，人类劳动的成本大致保持不变，而领先模型在这些任务上的表现持续提升、单位性能价格持续下降。
这一成本差异，把 RLHF 方法实验的市场开放给了此前被价格挡在门外的整个人群。

除价格之外，AI 反馈在性能上引入了与人类反馈不同的*权衡*，而这些仍在更广泛的文献中被研究。
AI 反馈在**评估**我们正在训练的语言模型这件事上，其作用要突出得多——因为它的低价格让它能用于各种大规模任务，而在那些任务上用人类数据的成本（或时间延迟）是不现实的。
这些话题彼此深度交织——即便在评估上，AI 反馈数据也永远不会完全取代人类数据；而且用于评估的 AI 反馈数量会远超训练，因为做评估的人远比训练模型的人多。

AI 反馈数据在哪些领域与应用上（即聊天、安全、推理、数学等）优于人类数据，尚未完全确立。
RLAIF 的一些早期工作表明 AI 反馈可以完全取代人类数据，把它吹捧为有效替代品 [@lee2023rlaif]，尤其是在仅以聊天任务评估时 [@cui2023ultrafeedback] [@yuan2025selfrewardinglanguagemodels]。
ChatGPT 之后研究 RLHF 的早期文献，评估套件很窄，聚焦于“在多种领域中充当有帮助助手的模型”的对齐（第 17 章进一步讨论）。
后来的工作给出了更细致的图景：在更广的评估集（例如包含一些推理任务）上，最优均衡是把一批有挑战性的数据点路由给人类做准确标注，而大多数数据送去做 AI 反馈 [@miranda2024hybrid] [@xu2025rlthf]。
尽管还没有研究专门聚焦 RLHF 中人类与 AI 反馈数据在更广领域上的配比，但有许多技术报告表明 RLHF 总体上能改善这一整套评估：有些用 DPO，如 Ai2 的 Tülu 3 [@lambert2024t] 与 Olmo 3 [@teamolmo2025olmo3]、Hugging Face 的 SmolLM 3 [@bakouch2025smollm3]；另一些用在线 RLHF 流水线，如 NVIDIA 混合使用来自 Scale AI 的人类偏好数据与基于 LLM 的反馈（通过 HelpSteer 系列工作 [@wang2024helpsteer] [@wang2024helpsteer2] [@wang2024helpsteer2p] [@wang2025helpsteer3]）：Nemotron Nano 3 [@nvidia2025nemotron3nano]、Nemotron-Cascade [@wang2025nemotron]，或 Llama-Nemotron 推理模型 [@bercovich2025llamanemotron]。

总体而言，尽管 AI 反馈及相关方法对这个领域显然极其有用，但人类数据显然并未被这些更便宜的替代品完全取代。
对此存在许多假说；但“人类数据是否能让模型在真实产品场景中获得更精细的控制，或支撑更新的训练方法（如角色训练——一组让模型性格可精确控制的新兴技术，见第 17 章）”这一点尚未被研究。
对刚起步的人来说，AI 反馈应当是首选尝试；但对规模扩大的流水线，最终过渡到引入人类反馈很可能是必然的。

**RLAIF 这个术语由 Anthropic 的工作《Constitutional AI: Harmlessness from AI Feedback》引入** [@bai2022constitutional]，这在 AI 社区引发了最初的困惑：论文标题里的两个方法（宪法式 AI 与 AI 反馈）是什么关系。
自 CAI 论文发布、RLAIF 被形式化以来，RLAIF 已成为后训练与 RLHF 文献中的默认方法——例子多到难以尽数。
这个关系应当这样理解：**CAI 是点燃 RLAIF 这一更广领域的那个范例。**

关于人类数据与 AI 反馈数据的差别，有一条经验法则：

1. **人类数据噪声高、偏差低。** 这意味着数据的采集与过滤可能更难，但一旦梳理得当，它会提供非常可靠的信号。
2. **合成偏好数据噪声低、偏差高。** 这意味着 AI 反馈数据更容易起步，但可能在模型上产生棘手的、非预期的二阶效应——而且这些效应会系统性地体现在数据中。

本书强调了许多学术结果，展示如何在 RLHF 工作流中替换进 AI 偏好数据并取得强劲的评估分数 [@miranda2024hybrid]；但更广的行业趋势显示，RLHF 的文献与更不透明的“最佳实践”之间存在脱节。
在整个行业中，人类数据常被视为一道实质性的壁垒与重大的技术优势。

### 为判断专门构建 LLM

随着 RLAIF 方法日益普及，许多人开始思考：**生成回答的模型，与生成评语或评分的模型，是否应当是同一个？**
具体来说，所用“LLM 作为评判者”的校准性受到了质疑。
若干工作表明 LLM 是不一致的评估者 [@wang2023large]，并且偏好自己的回答甚于其他模型的回答（被称为**自我偏好偏差**）[@panickssery2024llm]。

鉴于这些偏置，许多人问：解决方案是否应当是**专门训练一个模型**来做这项标注任务？
已经发布了多个模型，目标就是替代前沿模型作为数据标注工具，例如批评家模型 Shepherd [@wang2023shepherd] 与 CritiqueLLM [@ke2023critiquellm]，或类似 Auto-J [@li2023generative]、Prometheus [@kim2023prometheus]、Prometheus 2 [@kim2024prometheus]、Prometheus-Vision [@lee2024prometheus] 这类评估回答性能的模型；但它们在公开记录的训练配方中并未被广泛采用。
也有人发现，通过重复采样扩展推理 [@brown2024large] [@zhao2025sample] [@kalra2025verdict]、自我精炼 [@madaan2023self]，或锦标赛式排序 [@pace2024west]，能给出对真实判断更好的估计、或更高质量的偏好对。
其他校准方法让模型的生成能力与判断能力协同演化 [@wu2024meta]。
目前普遍接受的观点是：尽管偏置存在，但领先的语言模型为这项任务经过了大量训练——因为 AI 实验室的内部运营需要它，客户也大量使用它——所以一般**不需要自己训一个评判模型**，除非你的任务涉及大量未公开在互联网上的私有信息。

## 宪法式 AI

宪法式 AI（Constitutional AI，CAI）——Anthropic 用在 Claude 模型上的方法——是最早有记录的大规模、把合成数据用于 RLHF 训练的做法。
CAI 以两种方式生成合成数据：

1. 对指令微调数据做**批评与修订**，使其遵循一组原则，例如“这个回答是否在鼓励暴力？”或“这个回答是否真实？”。当模型为问题生成答案后，它拿答案去对照“宪法”里的原则清单，逐步精炼答案。然后模型在这个产出的数据集上微调。
2. 生成成对偏好数据：给定宪法中随机一条原则作为上下文，用语言模型回答“哪个补全更好”（类似于原则引导奖励模型的研究 [@sun2024salmon]）。随后 RLHF 就以常规方式在合成数据上进行——RLAIF 之名由此而来。

大体上，CAI 以第二部分（偏好数据）著称；但它为指令数据引入的方法，被后训练各处的通用数据过滤与合成数据生成方法所采用。

CAI 可以形式化如下。

通过采用一套人写的原则——他们称之为*宪法*——Bai 等 2022 用一个独立的 LLM 生成用于微调的人工偏好数据与指令数据 [@bai2022constitutional]。
宪法 $\mathcal{C}$ 是一组书面原则，指明在批评阶段应关注的特定方面。
指令数据的策展方式是反复采样一条原则 $c_i \in \mathcal{C}$，要求模型修改它对提示 $x$ 的最新输出 $y^i$，使其符合 $c_i$。
这就从用于批评的原则 $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$ 得到一系列指令变体 $\{y^0, y^1, \cdots, y^n\}$。
最终的数据点，是提示 $x$ 连同某个 $n$ 对应的最终补全 $y^n$。

偏好数据的构造方式类似但更简单：用 $\mathcal{C}$ 中原则的一个子集作为反馈模型的上下文。
反馈模型拿到提示 $x$、一组原则 $\{c_0, \cdots, c_n\}$，以及来自先前 RLHF 数据集、标记为答案 (A) 与 (B) 的两个补全 $y_0$ 与 $y_1$。
新数据点的生成方式是：让语言模型选出哪个输出 (A) 或 (B) 既质量更高、又更符合所述原则。
在较早的模型上，这可以通过让模型接着 `The answer is: ` 并观察 (A) 还是 (B) 的概率更高来实现；但如今更常见的做法是用一个会先解释推理、再给出选择的模型——通常称为一种**生成式奖励模型** [@mahan2024generative]。

### CAI 的延伸阅读

与宪法式 AI 相关的研究方向与扩展有很多，但其中少有被记录为 RLHF 与后训练配方上的明确改进。

- OpenAI 发布了 Model Spec [@openai2024modelspec]，一份陈述其模型预期行为的文档，并表示他们正在探索“让模型直接引用该文档”的对齐方法（可以看作 CAI 的近亲）。OpenAI 持续更新这份 spec，并用一种称为**审慎对齐**（Deliberative Alignment）的方法 [@guan2024deliberative] 训练其 o1 等推理模型，让模型在引用这些安全或行为政策的同时完成对齐。
- Anthropic 在其模型训练中继续使用 CAI，更新 Claude 所用的宪法 [@Anthropic2023ClaudesConstitution]，并实验“人群集体如何就模型原则达成共识”、以及外部团体自行制定原则再交给 Anthropic 训练模型时行为如何变化 [@ganguli2023]。
- 开源社区探索了把 CAI 复现到开放数据集上 [@Huang2024cai]，以及用它生成 LM 之间对话数据的探索 [@lambert2024self]。
- 其他工作用不同的优化方法配合原则驱动的偏好或反馈。
Sun 等 2023 [@sun2023principledriven] 把原则作为奖励模型的上下文，用来训练 Dromedary 模型 [@sun2024salmon]。
Glaese 等 2022 [@glaese2022improving] 用原则来提升 RLHF 过程中人类判断的准确性。
Liu 等 2025 [@liu2025inference] 训练一个奖励模型在推理时自行生成原则，并用它们给出最终分数。
Franken 等 2024 [@franken2024self] 把“遵循原则”表述为一个互信息最大化问题，预训练模型无需标签即可学习。

## Rubrics：面向具体提示的 AI 反馈

AI 反馈在训练中的作用，从 2024 年末到 2025 年持续增长，因为这个领域在寻找扩展“可验证奖励强化学习”的途径（见第 7 章）。
**评分准则**（rubric）的想法随之出现，作为一种为“没有明确可验证答案的提示”获取**近乎可验证**标准的方式。
这能让模型尝试为一个问题生成多个答案，并通过 RL 朝最好的答案更新。
这个想法与本章讨论的其他方法密切相关，很可能随着全行业 LLM 评判者与合成数据实践的改进而开始起作用。
如今，“以 rubric 作为奖励的 RL”已被确立能在科学推理、事实性等技能上带来有意义的提升 [@gunjal2025rubrics; @viswanathan2025checklists; @rezaei2025onlinerubrics; @liu2025openrubrics]。

下面是一个 rubric 示例及其对应提示 [@liu2025openrubrics]：
```text
**Prompt**: As a museum curator, can you suggest five obscure artifacts that would be perfect for a "Mysteries of the Ancient World" exhibit? Each artifact should come from a different culture and time period, with a brief description of their historical significance and mysterious origins. These artifacts should leave visitors wondering about the secrets and lost knowledge of our past. Thank you for your expertise in bringing this exhibit to life.

** Rubric**: 
1. The response includes exactly five distinct artifacts as requested. [Hard Rule] 
2. The response ensures each artifact originates from a different culture and time period. [Hard Rule] 
3. The response provides a brief description of each artifact's historical significance. [Hard Rule] 
4. The response provides a brief description of each artifact's mysterious origins or unexplained aspects. [Hard Rule] 
5. The response conveys a sense of intrigue and mystery that aligns with the theme of the exhibit. [Hard Rule] 
6. The response clearly and accurately communicates information in a well-organized and coherent manner. [Principle] 
7. The response demonstrates precision and clarity by avoiding unnecessary or irrelevant details. [Principle] 
8. The response uses informative and engaging language that stimulates curiosity and critical thinking. [Principle] 
9. The response shows thoughtful selection by ensuring each example contributes uniquely to the overall theme without redundancy. [Principle] 
10. The response maintains consistency in style and format to enhance readability and comprehension. [Principle]
```

`[Hard Rule]` 与 `[Principle]` 是特定标签，用来标示某条反馈的优先级。也可以用其他方式表示重要性，例如简单的优先级数字。

Rubric 生成通常**逐提示**地在训练数据上进行，这会在准备阶段累积可观的合成数据成本。
为缓解这一点，常见做法是为每个领域先套用一个通用 rubric 作为起点，再由一个监督语言模型为每个提示分配细粒度的 rubric 分数，以引导训练反馈。
下面是一个为科学任务生成 rubric 的示例提示 [@gunjal2025rubrics]：

```text
You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. 
Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question in one of these domains. 
Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. 
Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any key facts or explanations.

Total items:
- Choose 7-20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2-4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for. 

For example:
- Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential)
before the event equals the total mechanical energy after the event.
- Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic
energy plus initial potential energy equals final kinetic energy plus final potential energy.
- Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential
energy, to illustrate how energy shifts within the system.
- Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying
conservation of mechanical energy.

3. weight: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2.

Category guidance:
- Essential: Critical facts or safety checks; omission invalidates the response.
- Important: Key reasoning or completeness; strongly affects quality.
- Optional: Nice-to-have style or extra depth.
- Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
- When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc.
- If a clear conclusion is required (e.g. "The final answer is (B)"), include an Essential Criteria for it.
- If reasoning should precede the final answer, include an Important Criteria to that effect.
- If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys-title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
prefix, and no extra keys are allowed.
Now, given the question and reference_answer, generate the rubric as described. 
The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance.
```

另一个更简单的示例见 [@rezaei2025onlinerubrics]：

```text
SYSTEM:
You generate evaluation rubrics for grading an assistant's response to a user prompt.

Rubric design rules:
- Each criterion must be atomic (one thing), objective as possible, and written so a grader can apply it consistently.
- Avoid redundant/overlapping criteria; prefer criteria that partition different failure modes.
- Make criteria self-contained (don't rely on unstated context).
- Include an importance weight for each criterion.

Output format (JSON only):
{
  "initial_reasoning": "<brief rationale for what matters for this prompt>",
  "rubrics": [
    {
      "reasoning": "<why this criterion matters>",
      "criterion": "<clear, testable criterion>",
      "weight": <integer 1-10>
    },
    ...
  ]
}

USER:
User prompt:
{prompt}

Generate the rubric JSON now.
```

可以看到，这些提示可以非常详细，而且是针对具体训练设置调过的。

配合 RL 训练的 rubric，会继续在其早期应用（指令跟随 [@he2025advancedif]、深度研究 [@shao2025drtulu]、评估深度研究智能体 [@sharma2025researchrubrics]、长文生成 [@ruan2025expertlongbench]）之外继续演化。
