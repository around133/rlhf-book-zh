<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容与引用。
  代码块与英文提示词保留原文。
-->
---
prev-chapter: "正则化"
prev-url: "15-regularization"
page-title: 评估
search-title: "第 16 章：评估"
meta-description: "衡量 RLHF、后训练、奖励模型、开放式生成与模型行为的评估方法。"
next-chapter: "塑造模型性格与产品"
next-url: "17-product"
lectures:
  - video: "https://www.youtube.com/watch?v=dFafQmClYq4&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=19"
    label: "第 12 讲：前沿模型评估的演化"
---

# 评估

评估是一整套技术，用来理解本书所详述训练过程的质量与影响。
评估通常以**基准**（benchmark）的形式呈现（常见基准包括 MMLU、GPQA、SWE-bench、MATH 等），它们是离散的问题集或环境，用来测量模型的某个特定属性。
评估是一个不断演化的方法体系，所以这里呈现 RLHF 中近年评估的几轮“季节”，以及会延续到语言模型未来的共同主题。
理解语言模型评估（尤其是与后训练相关）的关键在于：**当前流行的评估范式，反映了当时流行的训练最佳实践与目标**。
尽管有挑战性的评估会把语言模型推向新领域，但大多数评估的设计初衷，是为新模型构建有用的信号。

从很多方面看，本章旨在呈现 RLHF 早期历史中各评估范式的“小品”，让读者理解其中的共同主题、细节与失效模式。

RLHF 与后训练的评估在其早期历史中经历了几个不同阶段：

1. **早期聊天阶段**：用 RLHF 或偏好微调训练的早期模型，其评估目标聚焦于捕捉模型的聊天表现，尤其是相对已知强模型（如 GPT-4）的表现。早期例子包括 MT-Bench [@zheng2023judging]、AlpacaEval [@dubois2024length] 与 Arena-Hard [@li2024crowdsourced]。这些基准用“LLM 作为评判者”取代人类评估者，用 GPT-4 这类模型给回答打分——一种高性价比地放大人类评估标准的做法（见第 12 章）。当时模型的评估范围很窄，而如今这些被视为“聊天”或“指令跟随”领域。
2. **多技能阶段**：随着时间推移，共识逐渐确立：RLHF 能改进的不只是聊天。例如 Tülu 的评估套件包含知识类任务（MMLU [@hendrycks2020measuring]、PopQA [@mallen2023llm_memorization]、TruthfulQA [@lin2021truthfulqa]）、推理（BigBenchHard [@suzgun2022challenging]、DROP [@dua2019drop]）、数学（MATH [@hendrycksmath2021]、GSM8K [@cobbe2021gsm8k]）、代码（HumanEval [@chen2021codex]、HumanEval+ [@evalplus]）、指令跟随 [@zhou2023instructionfollowingevaluationlargelanguage]，以及安全（多个评估的合成）。这反映了后训练被接纳为一个超越安全与聊天的多面向解决方案。
3. **推理与工具**：当前后训练时代由对高难度推理与工具使用问题的关注所定义。这包括难度高得多的知识密集型任务，如 GPQA Diamond [@rein2023gpqa] 与 Humanity's Last Exam [@phan2025hle]；复杂的软件工程任务，如 SWE-Bench+ [@aleithan2024swebenchplus] 与 LiveCodeBench [@jain2024livecodebench]；以及以近期 AIME 竞赛为代表的困难数学题。

在此之外，还会有新的领域演化出来。
随着 AI 愈发成为一个工业化领域，评估的激励结构正在改变，变得涉及多方利益相关者。
自 ChatGPT 发布以来，私有评估（如 Scale 排行榜 [@scale2024seal]）、社区驱动的评估（如 Arena [@chiang2024chatbot]）、以及第三方评估公司（如 Artificial Analysis 与 Epoch AI）大量涌现。
本章通篇会包含一些细节，映射这些评估是如何实现与理解的。

## 提示词的格式化

给语言模型**下提示**本身是一个简单动作，也相当自然；但它同时也被视为一门可以练习与精进的手艺或艺术 [@schulhoff2024prompt]。
提示是为语言模型组织信息与上下文的方式。
对常见交互，提示相对基础。
对高级场景，一个精心构造的提示会决定某个具体用例的成败。

在评估中，提示技巧对模型表现可能有实质性影响。
一些提示技巧——例如下面讨论的格式化——能让模型表现从 60% 掉到接近 0。
同样，改变提示也能帮助模型在训练中更好地学习。
口语上说，把模型提示得好，能带来“像在用未来模型”的主观体验，释放出正常使用之外的性能。

提示带来的收益通常小于“改进数据或训练算法”这类核心方向，但在最终产品中可能相当可观。
更重要的启示是：**训练一个强大的领先模型时，把它弄坏、让性能暴跌，比再多榨出一点性能要容易。**

用好现代语言模型的提示，可能涉及准备一整份报告让模型来回应（常常有数千 token 的生成文本）。
这种行为，是“语言模型性能如何被度量与理解”发生诸多变化之后的下游结果。

### 少样本提示与对数似然打分

早期语言模型只被当作智能自动补全使用。
为了以更开放的方式使用这些模型，人们会先给模型展示多个示例，再给一个不完整的短语作为提示。这被称为少样本或上下文学习 [@brown2020language]；当时还没有涉及指令微调或 RLHF。
以流行的评估为例，它长这样：

```text
# Few-Shot Prompt for a Question-Answering Task
You are a helpful assistant. Below are example interactions to guide your style:

### Example 1
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."

### Example 2
User: "Who wrote the novel '1984'?"
Assistant: "George Orwell wrote '1984.'"

# Now continue the conversation using the same style.
User: "Can you explain what a neural network is?"
Assistant:
```

这里评估一个答案有多种方式。如果我们考虑一个 MMLU 风格的问题，模型必须在多个答案中做选择：

```text
# Few-Shot Prompt

Below are examples of MMLU-style questions and answers:

### Example 1
Q: A right triangle has legs of lengths 3 and 4. What is the length of its hypotenuse?
Choices:
(A) 5
(B) 6
(C) 7
(D) 8

Correct Answer: (A)

### Example 2
Q: Which of the following is the chemical symbol for Sodium?
Choices:
(A) Na
(B) S
(C) N
(D) Ca

Correct Answer: (A)

### Now answer the new question in the same style:

Q: Which theorem states that if a function f is continuous on a closed interval [a,b], then f must attain both a maximum and a minimum on that interval?
Choices:
(A) The Mean Value Theorem
(B) The Intermediate Value Theorem
(C) The Extreme Value Theorem
(D) Rolle's Theorem

Correct Answer:
```

要让语言模型在这里给出答案，可以基于某些采样参数生成一个 token，看答案 A、B、C、D 是否正确（上面这种格式化由 [@robinson2023leveraging] 提出），也可以查看每个 token 的对数概率，若正确答案更可能就判定该题正确。

我们来细看这些评估细节。
前者在单次尝试时通常叫精确匹配，在聚合多次采样时叫多数投票（pass@k 是代码评估中对应的指标，用于测试功能正确性）；后者叫（条件）对数似然打分，其中条件就是提示。
核心差别是：从底层概率分布采样天然引入了随机性，而模型输出的 token 对数概率是静态的（忽略微小的数值差异）。

对数似然打分有两种可能的实现——第一，可以看字母 (A) 的概率，或答案 “The Mean Value Theorem.” 的概率。
两者都是可用的指标，但预测答案的字母比预测完整的、可能是多 token 的答案要简单得多。
对数似然打分在预训练评估中更常见——那里模型缺乏精确匹配所需的问答格式；而精确匹配在后训练中是标准做法 [@teamolmo2025olmo3]。

精确匹配有它自己的问题，例如要求刻板的格式后缀（如 `The answer is:`），或用正则表达式在生成文本中任意位置探测答案（如寻找 `(C)` 或答案字符串本身）。
如果评估格式与模型的生成方式不匹配，分数可能暴跌。
语言模型评估最好在“格式不构成瓶颈”时进行，这样才能测出模型的完整能力。
做到与格式无关的评估需要大量努力与试错，实践中相当罕见。

回到评估的历史。
无论上面用哪种设定，少样本提示的一个常见挑战是模型不遵循格式，而这会被计为错误答案。设计评估领域时，上下文里用多少个示例常被视为一个设计参数，范围从 3 到 8 个或更多。

### 思维链提示

在少样本提示的演化中，出现了“给模型提供思维链示例供其遵循”的想法。
其形式是让上下文示例包含写出来的推理，如下所示（后来被“显式提示模型生成推理步骤”所取代）[@wei2022chain]：

```text
# standard prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The answer is ...

# chain-of-thought prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria had 23 apples originally. They...
```

### 零样本指令跟随

随着时间推移、语言模型变强，它们演化到零样本评估，也就是“零样本学习器” [@wei2021finetuned]。
FLAN 表明：在特定任务上微调过的语言模型（作为现代指令微调的前身）可以泛化到它们没训练过的零样本问题上 [@wei2021finetuned]（T0 中也发现了类似结果 [@sanh2021multitask]）。
这就是指令微调（IFT）的出现——RLHF 与后训练的重要前身。
一个零样本问题长这样：

```text
User: "What is the capital of France?"
Assistant:
```

从 2022 年起，时间线开始纳入关键的早期 RLHF 工作，例如 InstructGPT。
伴随这些模型的核心能力与用例转变，是更加开放式的使用。
随着使用方式更开放，**从模型采样的评估**越来越流行，因为它映射了真实使用——技术上这可以叫基于生成（精确匹配）的评估，但没有那么明确的规范术语。
在这一时期直到 ChatGPT 之后的近几年，一些选择题评估仍在 RLHF 研究中使用，因为任何向通行实践的转变都需要相当长时间，通常是若干年才会展开（例如这类评估的做法是：把温度设为 0，采样字符 A、B、C 或 D）。

### 推理时代的评估提示

随着推理模型在 2024 年末、2025 年初兴起，模型行为的一个重大变化是：在每次回答之前加入一段很长的思维链（CoT）推理过程。
这些模型不再需要被提示那句经典的 “think step by step”（由 [@kojima2022large] 提出）。
评估实践的下一次演化，是**带思维链推理的基于生成（精确匹配）评估**（因此几乎总是在大于 0 的温度下、以获得最佳表现）。

例如在某些设定中，每个问题或类别都有专门设计的提示，用来从模型引出相应行为。
Tülu 3 是一篇早期的奠基性论文，详细列出了用于选择题 CoT 作答的一些提示 [@lambert2024t]。
下面是用于 MMLU 的一个提示示例——MMLU 正是从“单 token 答案采样”过渡到“长式 CoT + 精确匹配答案检查”的评估之一。

```text
Answer the following multiple-choice question by giving the correct answer letter in parentheses.
Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) ...

Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

这一点——尤其是当模型使用特殊格式把思考 token 与答案 token 分开时——使得评估范式必须做最近这一次重大更新。
评估正在转向：用思维链提示，让模型以生成式方式作答后再被测试。

### 智能体式评估的复杂性

随着模型走向智能体，评估范式正变得日益复杂。
系统提示与推理软件现在作为额外的层进入——主要通过一个 harness（承载框架）的中间软件——连同运行该软件的基础设施一起。
harness 是一个循环，包含提示与管理上下文的技能，例如压缩、工具、凭据等。
对智能体式评估，模型往往需要在**沙箱**中运行；沙箱是信息明确的、界限清晰的世界（例如完成任务所需的文件），并带有让评估可复现的规则（例如特定的工具定义）。
沙箱增加了运行模型的复杂度，因为通常除了推理用的 GPU，你还需要更多 CPU。
更多信息可参考 Florian Brand 的[这场演讲](https://www.youtube.com/watch?v=CGjuKIppZSs)、@fig:eval-components 中的系统图，或阅读这一时代最流行的评估 Terminal-Bench——其原始版本 [@tbench2025] 与更难的 Terminal-Bench 2.0 [@tbench2026]。

![运行一次现代智能体式评估的组件——每一个方框都会影响最终得分。图改编自 Florian Brand 的演讲 “LLM benchmarks in the era of agents”。](images/eval_components_tikz.png){#fig:eval-components .center data-dark-src=“images/eval_components_tikz-dark.png”}

## 为什么许多外部评估对比并不可靠

AI 公司模型公告里的语言模型评估，只能与其他新闻稿在很大误差棒下做比较——也就是说，稍微更好或更差的模型应当被视为等价——因为它们各自内部使用的评估流程没有跨模型受控、也没有被明确记录。
例如在 Olmo 3 项目中，作者发现推理模型时代的大多数后训练评估，在评估设置固定的情况下，标准差在 0.25 到 1.5 分之间 [@teamolmo2025olmo3]——而更大的分数变化可能来自使用不同的提示或采样参数。
实验室在训练期间对评估做“爬坡”（hillclimbing），让模型更有用；他们传统上混用训练集、开发集（即验证集）与留出评估集（即测试集）。
“爬坡”是一个口语说法，用来描述让模型在一组目标基准上逐步变好的做法。
对社区用来比较领先模型的公开评估，我们无法知道哪些被用于训练、哪些被留出测试。

随着评估分数成为企业营销方案的核心组件，公司内部的实现已经发生漂移。
有传言说主要 AI 实验室对 GSM8K 或 MATH 这类重要评估使用“自定义提示”。
这些做法演化得很快。

语言模型的评估栈之所以被认为带有营销色彩，是因为这些评估没有硬性的真值来源。
前沿实验室内部发生的是：评估套件正被调成适合其内部需求。
当结果被分享时，我们得到的是“某实验室为其模型得到的那些数字”，而不是那个函数的所有输入。
这些输入是极为敏感的配置，而且在 OpenAI、Meta、Anthropic 与 Google 各不相同。
即便是完全开放的评估标准，也很难保证可复现性。
**把精力集中在你自己的模型上，是接近可重复评估技术的唯一办法。**
这些营销背后有良好意图，尤其从技术团队的角度看。

多实验室评估对比中另一个混乱的来源，是评估比较里加入了推理时扩展。
推理时扩展表明模型可以通过在推理时使用更多 token 来提升表现。
因此，用推理时的总 token 数来控制评估分数很重要，但还不是通行实践。

取决于你的后训练数据如何格式化，模型在不同评估格式上会有显著差异。
例如两个流行的开放数学数据集 NuminaMath [@li2024numinamath] 与 MetaMath [@yu2023metamath]，由于答案格式的细微差异而互相冲突——Numina 把答案放在 `\boxed{XYZ}` 中，MetaMath 把答案放在 `The answer is: XYZ` 之后——同时在这两个上训练，效果可能比只用一个更差。
强模型被训练成能适应多种格式，但一般都有一个最擅长的格式。

最终，关于闭源模型的评估现状，我们剩下几个要点：

- 我们不知道、也未必拥有实验室正在爬坡的那些关键测试集，所以一些评估只是代理。
- 前沿模型的推理正因特殊系统提示、特殊 token 等变得更复杂，而我们不知道这如何影响评估；
- 我们不知道闭源评估在数值报告时所用的全部格式与细节。

所有这些动态，加上过去几年 AI 模型的飞速进展，造成了像 @fig:benchmark-saturation 那样著名的曲线——每个时代当红的基准都很快被攻克。
描述这种单个基准层面动态的常用术语是**饱和**（saturation）。
随着每个基准逼近 100%，模型的进展开始放缓，因为只剩下更难的（在很多情况下是标注错误的）数据点，这使它作为“训练进展度量（或两模型比较）”的可靠性下降。

![Epoch AI 的报告，展示主要 AI 评估如何随时间迅速饱和（饱和指某个基准达到满性能、模型不再有有意义的信号）。许可 CC-BY。](images/benchmark-performance.jpeg){#fig:benchmark-saturation}

## 实验室实际如何在内部用评估改进模型

对前沿语言模型的评估，今天既是一门科学，也同等是一门艺术；要精确规定不同团队如何使用评估来理解最前沿的语言模型，本身就需要另写一本书。

不同团队选择不同的评估来保持独立性（即让它们成为真正的测试集），但没人披露他们选了哪些。
例如流行的推理评估 MATH 与 GSM8K，它们的训练集里都有可以直接用来提升性能的提示。
用同分布的提示提升性能，与通过在通用数学数据上训练来泛化到这些任务，是完全不同的两回事。

事实上，这些*训练集*包含质量非常高的数据，模型会从在它们上面训练中受益。
如果这些公司**没有**把相应的评估当作追踪的核心指标，那么在评估集上训练可能是一个务实决策——因为高质量数据是模型开发的主要限制因素。

领先的 AI 实验室通过聚焦少数几个关键评估来做爬坡，最后报告核心公开集上的分数。要点是：**他们用于追踪进展的一些评估并不公开**，例如 GPT-4 报告中用于缩放预测交叉熵损失的数据集 [@achiam2023gpt]。

后训练评估在很大程度上依赖人类评估。
对生成式语言模型的人类评估会产生 Elo 排名（在早期 Anthropic 论文如宪法式 AI 中很流行），对奖励模型的人类评估则给出一致性。
这些也可以通过 A/B 测试窗口向用户投放两个不同模型来获得（见[偏好数据那一章](https://rlhfbook.com/c/11-preference-data)）。

他们选择聚焦的那一小撮评估，构成了评估与训练之间的紧密联系。
有一段时间，一个被聚焦的评估是 MMLU。
在推理模型出现期间，GPQA 因为社区对科学能力的关注上升而极其流行。
实验室会改变评估，使它们更适合自己的需求，例如 OpenAI 发布 SWE-bench Verified [@openai2024swebench]。
每个前沿实验室还自建或购买了更多公众无法访问的内部评估。

在内部改进评估对下游训练的关键作用，是**提升比较不同训练运行时的统计功效**。
通过改变评估，这些实验室降低了其优先信号上的噪声，从而做出更有依据的训练决策。

这一点又与现代语言模型训练栈中后训练的复杂度相互叠加。
今天评估语言模型需要相当量的 token 生成（而不只是看答案的对数概率），因此需要算力开销。
人们普遍接受前沿实验室会用小技巧在许多任务上提升表现——最常见的说法是为某些评估使用一次性提示。

## 数据污染

当前语言模型实践（即不限于 RLHF 与后训练）的一个重大问题，是有意或无意地把评估数据集中的数据用于训练。
这被称为*数据集污染*（数据集泄漏的一种形式），对应的规避做法叫*去污染*（decontamination）。
要对一个数据集去污染，就要在训练集与测试集上做搜索，寻找词/子词 token 上的 n-gram 重叠匹配，或固定长度字符子串匹配（例如 50 个字符）[@singh2024evaluation]。
数据被污染的途径很多，但最常见的是从网络上抓取多个阶段的训练数据。
基准常常被列在会被爬取的公开网络域名上，或者用户把问题投给模型，这些问题随后可能进入未来模型的候选训练数据。

例如，在为 Tülu 3 的评估套件做去污染时，作者发现流行的开放数据集被 RLHF 常用评估污染了 [@lambert2024t]。
这些重叠包括：UltraFeedback 与 TruthfulQA 的污染、Evol-CodeAlpaca 与 HumanEval 的污染、NuminaMath 与 MATH 的污染，以及 WildChat 与安全评估的污染。
这些是通过“训练提示到评估集中精确提示”的 8-gram 重叠发现的。

另一些情况下，模型被发现曾在与基准非常接近的数据上训练，例如保持数学题的措辞不变、只改数字；这会在后训练范式中导致异常行为，比如**用随机奖励做 RL 训练时基准分数却上升**——这是一个只有在模型存在某类数据污染时才应该提升性能的人为设定。
这种基座模型层面的污染——无法确切证明模型为何如此表现——已成为许多建立在 Qwen 2.5 与 Qwen 3 基座模型之上的早期 RLVR 工作的重大混淆变量 [@shao2025spurious] [@wu2025reasoning]。

为了理解那些不披露、也不开放训练数据的模型是否被污染，人们创建了基准的新版本，把原始问题做轻微扰动（例如针对 MATH 的 [@huang2025math]），以观察哪些模型是被训练来匹配原始格式或原始问题的。
在这些扰动基准上的高方差并不能确认污染（污染很难证明）。它更可能指示：模型是按某种特定格式训练的，而那未必迁移到真实世界表现。


## 工具

有许多开源评估工具可供选择。其中一些包括：

- 英国安全研究所的 Inspect AI [@inspectAI2024]；
- Hugging Face 的 LightEval [@fourrier2023lighteval]，它驱动了 Open LLM Leaderboard [@open-llm-leaderboard-v2]；
- EleutherAI 的 evaluation harness [@gao2023evalharness]，构建在其 GPT-Neo-X 模型的基础设施之上（其中包含一套良好的 GPT-3 时代评估设置与配置）[@gpt-neox-20b]；
- Ai2 基于 OLMES 的库 [@gu2024olmes]；
- 斯坦福基础模型研究中心的 HELM [@liang2023helm]；
- Mosaic（现为 Databricks）的 Eval Gauntlet [@mosaicml2024gauntlet]；等等。
