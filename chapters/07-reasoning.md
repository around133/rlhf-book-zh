<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
  提示词示例与模型输出保留原文。
-->
---
prev-chapter: "强化学习"
prev-url: "06-policy-gradients"
page-title: 推理与推理时扩展
search-title: "第 7 章：推理与推理时扩展"
meta-description: "后训练中的推理训练与推理时扩展，包括 RLVR 与思考模型。"
next-chapter: "直接对齐算法"
next-url: "08-direct-alignment"
lectures:
  - video: "https://www.youtube.com/watch?v=o4AB5xHIDdM&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=7"
    label: "第 5 讲：推理模型的崛起"
---

# 推理与推理时扩展

推理模型与推理时扩展，让语言模型的性能在 2024 年末、整个 2025 年以及未来，实现了巨大的跃升。
**推理时扩展**（inference-time scaling）指的是通过生成时投入更多算力来提升模型表现的能力——例如产出更长的推理链，或采样多个回答。
那些被训练成“回答前先充分思考”的语言模型，把这一特性利用得非常好。
这些模型用大量的可验证奖励强化学习（RLVR）训练 [@lambert2024t]，同时也仍然大量使用 RLHF。
本章回顾 AI 社区“对 RL 在语言模型中潜力的认识被彻底改写”这一过程，梳理 RLVR 的基本原理，点出关键工作，并指出未来几年将定义这一领域的主要争论。

## RLVR 的角色

先从一个典故说起。在 2016 年的神经信息处理系统大会（NeurIPS）上，Yann LeCun 首次提出他那个如今广为人知的“蛋糕”比喻，用来描述现代机器学习系统中的学习发生在哪里：

> 如果智能是一块蛋糕，那么蛋糕的主体是无监督学习，糖霜是监督学习，而蛋糕上的樱桃是强化学习（RL）。

这个比喻放在现代语言模型和近期后训练技术栈的变化上，基本已经完整了。
RLHF 是它的前奏，而用于推理模型（主要针对数学、代码和科学主题）的 RL，则是它的印证。在这个比喻里：

- 在海量互联网数据上的自监督学习构成了蛋糕的主体（若按 FLOPs 消耗的算力看尤其如此）；
- 后训练的起点——面向指令的监督微调（SFT）——把模型调向更窄的分布；
- 最后，“纯粹的”强化学习（RL）是顶上那颗樱桃。用来打造新一代“推理”或“思考”模型的规模化强化学习，就是这最后一步（同时也借助了 RLHF——如我们稍后所解释，它并不被视为经典 RL）。

这一点推理训练，催生了**思考模型**：它们把本书讨论的后训练技术（用于对齐偏好）与在可验证领域上的 RL 训练结合起来，从而大幅提升推理、编程、数学解题等能力。

这类模型的训练方法——可验证奖励的强化学习（RLVR）[@lambert2024t]——与 RLHF 非常相似，但它把奖励模型变成可选项，代之以一个打分函数：答案正确时返回正奖励，否则返回 0。

举例来说，对比一下 RLHF 与 RLVR 在“给回答打分”这件事上的差别。
RLHF 中，奖励模型必须评估主观品质：

> **提示**：解释经济学中的机会成本概念。
>
> **回答**：机会成本是你在做决策时放弃的次优选择的价值。例如，如果你花一小时学习而不是工作，机会成本就是你本可以挣到的工资……

给这个回答打分需要判断清晰度、准确性、完整性和有用性——这些品质都依赖学到的偏好，且没有确定的正确答案。

相比之下，RLVR 使用返回确定分数的验证函数。对数学而言：

> **提示**：所有小于 20 的质数之和是多少？
>
> **回答**：小于 20 的质数是 2、3、5、7、11、13、17 和 19。
> 相加：2 + 3 = 5，然后 5 + 5 = 10，然后 10 + 7 = 17，然后 17 + 11 = 28，然后 28 + 13 = 41，然后 41 + 17 = 58，最后 58 + 19 = 77。
> 答案是 $\boxed{77}$。
>
> **验证**：`extracted_answer == 77` → 奖励 = 1

`\boxed{}` 这个记号借自数学排版惯例，它让答案抽取变得直接——一条简单的正则表达式就能从回答里取出最终答案，无论模型是怎么得到它的。
注意还存在其他答案抽取方式，例如只用短语 “The answer is: ”（上面也展示了）、用 `<answer>` 这类特殊 token，或用 `####` 这类分隔符。

对代码生成而言，验证常常采取单元测试的形式：

> **提示**：写一个 Python 函数 `fib(n)`，返回第 n 个斐波那契数，其中 fib(0) = 0、fib(1) = 1。
>
> **回答**：
> def fib(n):
>     if n < 2:
>         return n
>     return fib(n - 1) + fib(n - 2)
>
> **验证（单元测试）**：
>
> assert fib(0) == 0   # base case
> assert fib(1) == 1   # base case
> assert fib(10) == 55 # larger value
> （全部测试通过 → 奖励 = 1）


单元测试是代码天然的验证函数：把模型的解答放到已知的输入—输出对上执行。
一种常见的打分形式是做简单的门控：如果所有断言通过，奖励为 1；任一失败则为 0。
其他设置会按通过的测试比例给部分分。
这两个例子里都不需要学到的奖励模型，而且大多数设置也确实不用（因为在这些领域模型对过度优化有较强的抵抗力）；不过也可以用奖励的线性组合把奖励模型加进来。

RLVR 背后的想法在 RL 文献中并不新鲜——“依据答案是否正确做梯度更新”几乎就是强化学习的教科书定义。
把它应用到语言模型上的创新，主要在于**如何在保持被微调模型的强大通用能力的前提下应用它**。在此之内，语言模型文献中还有许多相关想法：让模型从“关于答案正确性的反馈”中学习。

最初，在提出“可验证奖励的 RL”（RLVR）这一术语的那项工作中 [@lambert2024t]，这个方法本来要叫“真值奖励的 RL”（RLGT）。
但 RLVR 与“仅从真值答案学习”有微妙差别。
在数学这类领域，有唯一的真值答案可用于验证解答，如上所示。
而在代码生成或精确指令跟随这类领域，即使正确答案不止一个、没有唯一真值，答案仍可用检查函数（例如单元测试）来验证。
RLVR 进展的核心，是拥有多样且有深度的这类**可验证问题**——即便确切解并不事先已知。

![以 RL 反馈循环形式呈现的 RLVR。用验证函数取代奖励模型。](images/rlvr-system.png){#fig:rlvr data-dark-src=“images/rlvr-system-dark.png”}

最早成功部署这类训练的模型是 OpenAI 的 o1 [@openai2024o1] 与开放权重模型 DeepSeek R1 [@guo2025deepseek]。
不久之后，整个 AI 行业都把这一训练过程和模型风格提上了优先位置。
这里的核心变化与其说是“这种 RL 设定全新”，不如说是**训练阶段的重新分配、以及不同行为优先级的调整**。
推理模型带来了一个“扩展 RL 训练成为预期动作”的时代。

至于这类模型表现出什么样的行为，看下面这个例子：用 DeepSeek V3 0325 与他们的推理模型 DeepSeek R1 回答 `Write me a short poem about a goldfish`。
DeepSeek V3 非常直接：

> **回答**：Goldfish Dreams
>
> In a bowl of glass so clear and round,
> A tiny goldfish swims without a sound.
> With scales that gleam like sunlight's kiss,
> He drifts along in watery bliss.
>
> A flick of fin, a flash so bright,
> Chasing bubbles in the light.
> Does he dream of oceans wide?
> Or is his little world enough---his pride?

而 DeepSeek R1 的体验不同。
它会先生成大量思考 token 来构思回答，然后再回复用户。
它的回答看起来是这样：

> **回答**：`<thinking>` Okay, the user wants a short poem about a goldfish. Let me start by brainstorming some goldfish-related imagery. Goldfish are often in bowls, right? So maybe start with the setting. Words like “gilded bowl” or “crystal globe” could work.
>
> Next, think about the movement of the goldfish. They swim smoothly, so words like “dart”, “glide”, or “swirl” come to mind. Maybe describe their fins as “silken” to emphasize delicacy.
>
> Then, consider the environment around the goldfish. Sunlight through water creates shimmering effects. Words like “sunbeams dance” or “ripples spin” might capture that. Including elements like castles or pebbles adds a playful touch.
>
> Themes? Maybe the contrast between the goldfish's small world and its perception of freedom. The bowl as a universe, tiny but complete. Ending with a sense of mystery or silent observation could give depth. Need to keep it concise, maybe four stanzas with a rhyme scheme. Let me check the rhythm and flow. Avoid clichés but keep it accessible. Alright, let's piece it together. `</thinking>`
>
> Gilded Dreams in Liquid Glass
>
> A gilded bowl, a crystal globe,
> Where tiny tides in silence flow---
> A flash of orange, a whispered robe,
> You dart through sunlit indigo.
>
> Your world is small, yet boundless too---
> Each pebble, castle, weed unfurled.
> The surface bends the sky to blue,
> And ripples spin your liquid world.
>
> You glide through myths we'll never know,
> A silent monarch, veiled in mist.
> Do dreams drift where the bubbles go?
> Or memories in amethyst?
>
> We watch you trace your endless sphere---
> A universe within a tear.

`<thinking>` 标签里的 token 就是模型的推理过程。
对更复杂的问题，推理阶段可能先耗掉数千个 token 才产出答案。
所以长上下文语言模型是高级推理行为的前提，但这不是本章的重点。

理解*这套训练如何起作用*的核心直觉是：对给定模型，反复执行下面的循环：

1. 对多个问题采样多个答案；
2. 朝着**正确**的答案做梯度步；
3. 重复，反复回到同一批数据。

值得注意的是，这个极其简单的方法（在数据分布精心设计、训练基础设施稳定的前提下）通过一遍又一遍地重温同样的问题，帮助模型学习。
更值得注意的是：在这些训练问题上的提升，会泛化到模型从未见过的问题、（部分）领域上！

这个简单的方法让模型能在行为空间中做轻度搜索，而 RL 算法会提高那些与正确答案相关的行为的概率。

## 新一代推理模型的来路

下面梳理导致 2025 年推理模型爆发的高层趋势。

### 为什么 RL 现在奏效了？

尽管有非常多“RL 还不管用”的观点 [@irpan2018deep]，也有论文详细记录了 RL 的可复现性问题 [@henderson2018deep]，但这个领域跨过了这些障碍，找到了高影响力的应用。
有些在本书中已有涉及，例如 ChatGPT 的 RLHF 与 DeepSeek R1 的 RLVR；但还有许多其他应用，包括改进芯片设计 [@mirhoseini2020chip]、掌握电子游戏 [@schrittwieser2020mastering]、自动驾驶 [@cusumano2025robust] 等等。
语言模型上以 RL 为核心的训练之所以能起飞，说明这一研究领域在许多根本问题上取得了进展，包括：

- **RL 的稳定性问题是可以解决的**：在其整个发展史上，限制 RL 被采用的一直是稳定性。它体现在两方面。第一，学习本身可能反复无常、并非总能奏效。第二，训练本身以比标准语言模型训练更脆弱、更容易出现损失尖峰和崩溃而著称。如今无数新模型都在预训练基座之上使用这种带可验证奖励的 RL 训练，学术界也大量跟进。**RL 的技术门槛正处于历史最低点。**

- **开源版本已经“存在”**：已经有许多工具可用于以 RLVR 及相关技术训练语言模型。
例如 TRL [@vonwerra2022trl]、Open Instruct [@lambert2024t]、veRL [@sheng2024hybridflow] 和 OpenRLHF [@hu2024openrlhf]，其中许多建立在 RLHF 与后训练早期阶段的优化之上。工具的可获得性正在支撑起一个庞大且加速增长的研究体量。

多项资料表明，面向推理的 RL 训练只有在领先模型上才可行，且大约从 2024 年起的模型才具备这一条件——这说明在推理训练成为可能之前，模型需要达到一定的底层能力水平。

### RL 训练与推理时扩展

用强化学习训练来激发推理行为、提升可验证领域上的表现，与**推理时扩展**的想法紧密相连。
推理时扩展（也叫测试时扩展）是指一类方法：在推理阶段投入更多算力，以便在下游任务上表现更好。
在 DeepSeek R1 与 OpenAI o1 发布之前，推理时扩展的方法就已被研究——而这两个模型都极大地推动了人们对 RL 训练的投入。
例子包括值引导采样 [@liu2023don]，以及带答案抽取的重复随机采样 [@brown2024large]。
除此之外，推理时扩展还能改进思维链推理之外的更多 AI 训练方法，例如使用会深入权衡各个选项的奖励模型 [@ankner2024critique] [@liu2025inference]。

RL 训练是通往“推理时扩展规律”的一条捷径；但长期来看，我们将拥有更多方法来激发所需的推理时权衡，以获得最佳性能。
用 RL 大量训练模型，往往能让它每次回答生成更多 token，而这一增长与下游性能提升高度相关（尽管序列变长是默认现象，也有研究专门探索**不依赖**这种推理时扩展来提升性能）。
这与早期 RLHF 系统中的长度偏置形成了重要对照 [@singhal2023long]——那时人类偏好训练有个副作用：为了在偏好排序上取得边际收益而拉长了平均回复长度。

除了核心的 RL 训练模型之外，还有许多方法正在被探索，以继续推进推理与推理时算力的极限。
由于它们演化太快，大多超出本书范围；这些方法包括：通过指令微调把大型 RL 训练模型的推理行为蒸馏到小模型 [@muennighoff2025s1]、组合更多次推理调用 [@chen2024more] 等。
这里要紧的是**下游性能与生成 token 数增长之间的相关性**——否则就只是白白浪费能量。


### RLVR 的未来（超越推理）

在许多领域，这些新形态的 RLVR 更贴合开发者的目标：它关注**性能**而非**行为**。
标准的微调 API 通常使用 LoRA 这类参数高效微调方法（Low-Rank Adaptation，一种只训练少量新增矩阵、而非全部模型权重的参数高效方法，也叫参数高效微调 PEFT），配合对指令的监督微调。
开发者传入提示与补全，模型通过更新参数去匹配这些补全，从而在你的数据特征在模型生成中变得更常见。

而 RLVR 关注的是**匹配答案**。
给定查询与正确答案，RLVR 帮助模型学会产出正确答案。
标准指令微调通常只对数据做 1 到 2 个 epoch 的损失更新，而 RLVR 之所以得名，是因为它会对同样这几个数据点做数百到数千个 epoch，给模型时间去学习新行为。
这可以理解为：把基座模型版本中偶尔才奏效的正向行为，经 RLVR 之后强化为稳健的行为。

**语言模型 RL 训练的范围还在持续扩大**：从基础科学层面看，o1 与 R1 带来的最大启示是——我们有了更多训练语言模型、使其具备潜在有价值行为的手段。
对研究者和工程师敞开的大门越多，我们对 AI 总体发展轨迹就该越乐观。


## 理解推理训练方法

对推理的投入，引发了“如何训练模型遵循人类指令”这门手艺的重大演化。
这些流程仍然使用前面章节讨论过的常见组件（如第 3 章概述 DeepSeek R1 流程时所述），包括指令微调、人类反馈强化学习和可验证奖励的强化学习（RLVR）。
核心变化是**使用多得多的 RLVR，并以不同顺序施加其他训练技术**——传统上，对一个推理模型而言，核心训练步骤要么是一次大规模 RL 运行，要么是在另一个已接受大量 RLVR 训练之模型的*输出*上做大规模指令微调（即蒸馏）。

### OpenAI o1 与 DeepSeek R1 之前的推理研究

在推理模型起飞之前，人们已投入大量努力去理解如何训练语言模型、使其在可验证领域上更强。
下面这些工作与后来方法的主要差别在于：它们的方法论没有扩展到 DeepSeek R1 及其后继模型那样的规模，或者它们产出的模型为换取更强的数学或编程能力而在整体性能上做出了牺牲。
这里收录背后的想法与动机，是为了更完整地呈现推理模型是如何在这一图景中出现的。

最早在可验证领域训练语言模型的努力，包括自教推理器（STaR）系列工作 [@zelikman2022star] [@Zelikman2024QuietSTaRLM] 和 TRICE [@hoffman2023training]——两者都在 2022 与 2023 年间使用真值奖励信号来鼓励模型进行思维链推理。
STaR 实际上近似了策略梯度算法，但实践中样本过滤方式不同，并用交叉熵度量替代对数概率；而 Quiet-STaR 用一个与近期推理模型高度相关的想法扩展了它：让模型在尝试回答可验证问题**之前**先生成 token（这有助于训练表现）。
TRICE [@hoffman2023training] 同样通过生成推理轨迹、再用一种受马尔可夫链蒙特卡洛启发的期望最大化算法来优化，从而改进推理。
VinePPO [@VinePPO] 紧随其后，其设定更接近现代推理模型。
VinePPO 使用基于 PPO 的算法、以数学题正确性作为二元奖励，在 GSM8K 与 MATH 上训练。
在 OpenAI o1 与 DeepSeek R1 之前的其他工作，则用代码执行作为训练反馈信号 [@gehring2024rlefgroundingcodellms]、[@xu2024dpo]，或用验证器进行定理证明（这里称为“验证器反馈强化学习”，RLVF）[@amit2024models]。
Tülu 3 在这些方法上做了扩展：用一个简单的 PPO 训练器对答案正确的补全给予奖励——最重要的是，同时保持在广泛评估套件上的整体性能。
Tülu 3 的二元奖励与现代推理训练技术，可以与 STaR 的迭代式方法、或 Quiet-STaR 的对数似然奖励形成对照。

### 早期推理模型

下表汇总了 DeepSeek R1 之后那些奠基性的推理研究技术报告——其中一部分附有开放数据与模型权重。

| 日期        | 名称                        | 一句话概括                                                                  | 开放权重 | 开放数据 |
|-------------|----------------------------|-----------------------------------------------------------------------|--------------|-----------|
| 2025-01-22  | DeepSeek R1 [@guo2025deepseek]             | 基于 RL 的 DeepSeek 升级版，数学与代码推理解题能力大幅提升      |  是      | 否   |
| 2025-01-22  | Kimi 1.5 [@team2025kimi]                  | 在中英文数据上扩展 PPO/GRPO；AIME 数学表现强劲            | 否           | 否        |
| 2025-03-31  | Open-Reasoner-Zero [@hu2025openreasonerzero]   | 对基座模型 RL 的完全开放复现      |  是      |  是   |
| 2025-04-10  | Seed-Thinking 1.5 [@seed2025seed]         | 字节跳动的 RL 流水线，带动态 CoT 门控                         | 是     | 否   |
| 2025-04-30  | Phi-4 Reasoning [@abdin2025phi4]          | 14B 模型；细致的 SFT→RL；擅长 STEM 推理                   | 是      | 否   |
| 2025-05-02  | Llama-Nemotron [@bercovich2025llamanemotron]   | 多种规模的“推理开关”模型                 |  是      |  是   |
| 2025-05-12  | INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained] | 首个有公开记录的全球去中心化 RL 训练                     |  是      |  是   |
| 2025-05-12  | Xiaomi MiMo [@xia2025mimo]                | 从预训练到后训练的端到端推理流程              | 是          | 否       |
| 2025-05-14  | Qwen 3 [@yang2025qwen3]                   | 把类似 R1 的流程应用到新模型上                    |  是      | 否   |
| 2025-05-21  | Hunyuan-TurboS [@liu2025hunyuan]          | Mamba-Transformer MoE，自适应长/短 CoT                        | 否           | 否        |
| 2025-05-28  | Skywork OR-1 [@he2025skyworkor1]          | 避免熵崩塌的 RL 流程；在 AIME 上超过 DeepSeek           |  是      |  是   |
| 2025-06-04  | Xiaomi MiMo VL [@coreteam2025mimovltechnicalreport]                | 把推理流程端到端地扩展到包含多模态任务              | 是          | 否        |
| 2025-06-04  | OpenThoughts [@guha2025openthoughts]      | 从 QwQ-32B 蒸馏而来的公开 120 万条指令数据集                    |  是      |  是   |
| 2025-06-10  | Magistral [@mistral2025magistral]         | 在 Mistral 3 上做纯 RL；多语言 CoT；小模型开源      |  是| 否        |
| 2025-06-16 | MiniMax-M1 [@minimax2025minimax_m1] | 开放权重 456B MoE 混合/Lightning Attention 推理模型；100 万上下文；用 CISPO 做 RL；发布 4 万/8 万思考预算的检查点 | 是 | 否 |
| 2025-07-10 | Kimi K2 [@kimiteam2025kimik2]                            | 1T MoE（32B 激活），用 MuonClip（QK-clip）保证稳定；15.5T token 预训练无损失尖峰；多阶段后训练含智能体数据合成 + 联合 RL；发布基座与后训练检查点。                               | 是          | 否         |
| 2025-07-28 | GLM-4.5 [@zeng2025glm45] | 开放权重 355B-A32B MoE“ARC”模型，含思考/非思考模式；23T token 多阶段训练 + 专家迭代与 RL 的后训练；发布 GLM-4.5 与 GLM-4.5-Air（MIT）。 | 是 | 否 |
| 2025-08-20 | Nemotron Nano 2 [@nvidia2025nemotronnano2]               | 用于长“思考轨迹”的混合 Mamba-Transformer；20T token FP8 预训练后压缩/蒸馏；明确发布多个检查点以及预训练/后训练数据集的“大部分”。                                       | 是          | 是（大部分） |
| 2025-09-09 | K2-Think [@llm3602025k2think]                            | 参数高效的数学推理系统：32B 开放权重模型 + 测试时扩展配方；按发布材料定位为完全开放（含训练数据/代码）。                                                                       | 是          | 是        |
| 2025-09-23 | LongCat-Flash-Thinking [@mlcteam2025longcat]             | 560B MoE 推理模型；报告明确给出从长 CoT 冷启动到大规模 RL 的分阶段配方；开源发布。                                                                                                             | 是          | 否        |
| 2025-10-21 | Ring-1T [@ringteam2025everystepevolves]                  | 万亿级“思考模型”，聚焦 RL 扩展；报告阐述了 1T 规模下扩展 RL 的瓶颈与解法，并发布开放模型。                                                                                                             | 是          | 否        |
| 2025-11-20 | Olmo 3 Think [@teamolmo2025olmo3]         | 完全开放的“模型流”发布：报告了完整生命周期（各阶段、检查点与数据点），并把 Olmo 3 Think 32B 定位为旗舰级开放思考模型。                                        | 是          | 是        |
| 2025-12-02 | DeepSeek V3.2 [@deepseekai2025v32]                       | 开放权重 MoE 的前沿推进，报告突出注意力效率改动、RL 框架升级，以及面向智能体/推理性能的数据合成。                                                                             | 是          | 否        |
| 2025-12-05 | K2-V2 [@liu2025k2] | 70B 稠密“360 开放”模型，从零训练；仅用 SFT 的三档努力后训练实现可控思考。 | 是 | 是 |
| 2025-12-15 | Nemotron 3 Nano [@nvidia2025nemotron3nano]               | 30B-A3B MoE 混合 Mamba-Transformer；25T token 预训练并包含 SFT + 大规模 RL；明确说明发布权重 + 配方/代码 + 大部分训练数据。                                                                      | 是          | 是（大部分） |
| 2025-12-16 | MiMo-V2-Flash [@mimo2025flash] | 309B MoE（15B 激活）为速度优化：混合 SWA/GA 注意力（5:1，128 token 窗口）+ 轻量 MTP；27T token FP8 预训练；用 MOPD + 大规模智能体 RL 做推理/编程的后训练。 | 是 | 否 |
Table: 2025 年（RLHF 推理时扩展大规模兴起的第一年）值得注意的推理模型技术报告汇总。 {#tbl:reasoning_list}


### 训练推理模型的常见实践

本节详述训练推理模型时，为最大化性能而安排训练阶段顺序、改造数据的常见方法。

注意这些论文可能用了某个列出的技术却没有提及，而其他论文提到了；所以这些例子只是已知实现的子集，可作为参考，但不应视为对“最优配方是什么”的最终宣判。

- **离线难度过滤**：RLVR 的一个核心直觉是——模型只能从**有梯度**的样本中学习。如果 RLVR 的起点模型对一个问题的正确率是 100% 或 0%，那么该提示下不同补全之间就没有梯度（即所有策略在策略梯度算法看来都一样）。许多模型在开始大规模 RL 之前做难度过滤，把训练问题限制在起点模型正确率只有 20–80% 的那些。这类数据通过对训练集中每个提示采样 N 个（例如 16 个）补全、验证其中正确的比例来收集。Seed-Thinking 1.5、Open Reasoner Zero、Phi-4、INTELLECT-2、MiMo RL、Skywork OR-1 等都采用了这类做法。
- **批内在线过滤**（或贯穿训练的难度课程）：为配合离线过滤找到合适的问题，另一个主要问题是——**学习过程中应以什么顺序把问题呈现给模型？** 为处理这一点，许多模型使用批内问题的在线过滤、预构建的课程/数据调度器、把更难的问题留到训练后期，或其他提升长期稳定性的做法。Kimi 1.5、Magistral、Llama-Nemotron、INTELLECT-2、MiMo-RL、Hunyuan-TurboS 等都用了相关思路。
- **去掉 KL 惩罚**：随着推理模型的 RL 运行长度（无论以总 GPU 小时、FLOPS 还是 RL 步数衡量）相对 RLHF 训练大幅增加、并且奖励函数变得不那么容易被过度优化，许多模型去掉了“约束 RL 学到策略与训练起点基座模型相似”的 KL 惩罚。这让模型能在训练中做更多探索。RAGEN [@wang2025ragenunderstandingselfevolutionllm]、Magistral、OpenReasonerZero、Skywork OR-1 等都用了这一点。
- **放宽策略梯度裁剪**：GRPO 的新变体（如 DAPO [@yu2025dapo]）对 GRPO（或 PPO）中的双边裁剪目标提出修改，以支持更好的探索。也有研究表明，当奖励并不完美时，裁剪可能造成**虚假的学习信号** [@shao2025spurious]。RAGEN、Magistral、INTELLECT-2 等使用了这种“每个梯度方向用不同范围”的双边裁剪。
- **异策略数据（或完全异步更新）**：随着用 RL 解题所需的补全长度因问题变难而急剧增长（尤其是回复长度的*方差*——常出现极长的离群值），RL 运行中的算力可能闲置。为解决这一点，训练正在转向异步更新，或改变把问题排进批次的方式以提升整体吞吐。Seed-Thinking 1.5、INTELLECT-2 等使用了部分到完全的异步（异策略）数据。
- **额外的格式奖励**：为了让推理过程可预测，许多模型加入小的奖励，确保模型在给出答案前遵循 `<think>...</think>` 之类的正确格式。DeepSeek R1、OpenReasonerZero、Magistral、Skywork OR-1 等都用了这一点。
- **语言一致性奖励**：与格式奖励类似，一些多语言推理模型使用语言一致性奖励，优先奖励那些推理过程中不切换语言的模型（以获得更好、更可预测的用户体验）。包括 DeepSeek R1、Magistral 等。
- **长度惩罚**：许多模型在 RL 训练中使用不同形式的长度惩罚，或用于随时间稳定学习过程，或用于缓解在难题上的过度思考。例如 Kimi 1.5 逐步延长目标长度以对抗过度思考（同时保证在难度课程上训练准确率保持较高），而 INTELLECT-2 全程使用一个较小的长度惩罚。逐步延长训练序列长度之所以能缓解过度思考，是因为它先迫使模型在思考预算更受限的领域里高效推理，然后再过渡到更长的训练，让模型能把那些行为高效地用在更复杂的问题上。其他做法包括超长过滤等相关实现以提升吞吐。
- **损失归一化**：关于原始 GRPO 算法中按组归一化的项可能引入长度或难度偏置，已有一些讨论（见策略梯度那一章、或 [@liu2025understanding]）。因此一些模型（如 Magistral 或 MiMo）选择在批次层面而非组层面归一化损失或优势。
- **并行的测试时算力扩展**：把多个并行、独立采样的 rollout 的答案结合起来，可以比只用单个 rollout 的答案取得显著提升。最朴素的并行测试时算力扩展（DeepSeek-R1、Phi-4 等都这样做）是取多数 rollout 给出的答案作为最终答案。更高级的技术是用一个专门训练的打分模型，从并行 rollout 的答案中挑出最好的那个。截至 2026 年，这一技术在公开的、有文档的推理模型配方中还不常见，但 Claude 4 的发布公告中提到了它 [@anthropic2025claude4]，DeepSeek-GRM 也使用了它 [@liu2025inference]。

除了这些常见技术之外，还有不少关于“如何训练出有用的推理模型、同时不牺牲周边能力”的常见发现：

- **纯文本推理能提升多模态表现**：Magistral、MiMo-VL 等发现，先训练一个多模态模型，再在多模态训练之后做纯文本推理训练，可以*提升*最终模型的多模态表现。
- **用系统提示切换推理**（或长度控制）：Llama-Nemotron、Nemotron Nano、Qwen 3、SmolLM 3 等使用特定的系统提示（可能配合长度受控的 RL 训练 [@aggarwal2025l1]），让用户能开关思考长度。其他开放模型——如 OpenAI 的 gpt-oss 与 LLM360 的 K2-V2 [@liu2025k2]——在系统提示中采用“低—中—高”三档推理强度，但针对这类行为的训练方法文档化程度较低。

## 展望

推理模型的图景正在以近年 AI 研究中罕见的速度演化，这里列出的一些常见实践注定会被新技术取代。

目前有若干努力正在系统性地理解“是什么让推理训练奏效”。
Olmo 3 Think [@teamolmo2025olmo3] 是对一个推理模型完整训练生命周期最全面的公开记录，为研究社区提供了各阶段的检查点与数据，并以一次在 220 张 GPU 上持续近 4 周的训练收尾。
类似地，关于理解 RL 在推理上的扩展性质的工作 [@khatri2025art]，正在把此前只存在于从业者直觉中的“算力、数据与性能之间关系”形式化。

有一点仍然清楚：**强化学习已经从那块蛋糕上的“樱桃”，毕业成了前沿模型训练的承重结构。**
本章围绕 RLVR 的那些具体技术——难度过滤、格式奖励等等——并不是最终答案，但它们代表了当前这个领域对“如何从语言模型中激发推理”的最佳理解。
下一代方法很可能看起来不一样，但它们会建立在这里奠定的基础之上。
