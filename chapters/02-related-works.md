<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
-->
---
prev-chapter: "引言"
prev-url: "01-introduction"
page-title: RLHF 简史
search-title: "第 2 章：RLHF 简史"
meta-description: "支撑 RLHF、奖励建模、偏好学习与语言模型后训练的关键论文与历史节点。"
next-chapter: "训练总览"
next-url: "03-training-overview"
lectures:
  - video: "https://www.youtube.com/watch?v=MMDNaeIFVy8&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "第 0 讲：前置知识"
  - video: "https://www.youtube.com/watch?v=o6l6tJQgUg4&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "第 1 讲：总览（第 1–3 章）"
---

# RLHF 简史

RLHF 及其相关方法都非常年轻。
我们梳理这段历史，是为了说明这些流程被正式确立的时间有多近，以及其中有多少内容仍停留在学术文献里。
借此我们想强调：RLHF 正在飞速演化。所以本章为全书定一个基调——本书对某些方法会持保留态度，并预期某些细节可能围绕少数核心实践发生变化。
除此之外，这里列出的论文与方法，也解释了 RLHF 流程中许多组件为何是今天这个样子；因为其中一些奠基性论文所针对的应用，与现代语言模型完全不同。

本章详述把 RLHF 领域带到今天这一步的关键论文与项目。
它不是对 RLHF 及相关领域的全面综述，而是一个起点，讲述我们如何走到今天。
它有意识地聚焦于导向 ChatGPT 的近期工作。
RL 文献中关于“从偏好中学习”的研究还有很多 [@wirth2017survey]。
如需更完整的清单，应当查阅专门的综述论文 [@kaufmann2023survey]、[@casper2023open]。

![本章讨论的 RLHF 关键发展时间线：从早期“从偏好中做 RL”的工作，到 RLHF 被大语言模型采用。](images/rlhf_timeline.png){#fig:rlhf_timeline data-dark-src=“images/rlhf_timeline-dark.png”}

## 起源到 2018：从偏好中做 RL

这个领域近来随着深度强化学习的发展而流行起来，并已扩展为许多大型科技公司对 LLM 应用的更广泛研究。
不过，今天使用的许多技术，与早期“从偏好中做 RL”文献的核心技术有着很深的渊源。

最早提出类似现代 RLHF 思路的论文之一是 *TAMER*。
*TAMER: Training an Agent Manually via Evaluative Reinforcement* 提出的做法是：由人类反复给智能体的动作打分，据此学习一个奖励模型，再用它来学习动作策略 [@knox2008tamer]。
同期或不久之后的其他工作提出了 actor-critic 算法 COACH，它用人类反馈（正面和负面）来调整优势函数 [@macglashan2017interactive]。

最重要的参考文献是 Christiano 等 2017 年的工作，它把 RLHF 应用于 Atari 游戏中的智能体轨迹之间的偏好 [@christiano2017deep]。
这项引入 RLHF 的工作，紧跟在 DeepMind 关于深度 Q 网络（DQN）的奠基性强化学习工作之后——后者表明 RL 智能体可以从零学会玩流行的电子游戏。
这项工作表明：在某些领域，让人类在轨迹之间做选择，比直接与环境交互更有效。它借助了一些巧妙的条件设定，但无论如何都令人印象深刻。

![Christiano 等（2017）提出的核心 RLHF 循环：奖励预测器由轨迹片段的比较异步训练，智能体则最大化预测奖励。](images/rlhf_schematic.png){#fig:rlhf_schematic width=66% data-dark-src=“images/rlhf_schematic-dark.png”}

这一方法后来被更直接的奖励建模工作所扩展 [@ibarz2018reward]；而早期 RLHF 工作中对深度学习的采用，以一年后把 TAMER 扩展到神经网络模型为标志 [@warnell2018deep]。

这个时代开始转向：奖励模型作为一种通用概念被提出，用来研究“对齐”，而不仅仅是解决 RL 问题的工具 [@leike2018scalable]。

## 2019 到 2022：在语言模型上做人类偏好强化学习

人类反馈强化学习（早期也常被称为“人类偏好强化学习”）很快被越来越多转向规模化大语言模型的 AI 实验室采纳。
这项工作很大一部分始于 2019 年的 GPT-2 与 2020 年的 GPT-3 之间。
2019 年最早的工作 *Fine-Tuning Language Models from Human Preferences* 与现代 RLHF 工作、以及本书将要讲的内容，有许多惊人的相似之处 [@ziegler2019fine]。
许多规范术语——如学习奖励模型、KL 距离、反馈示意图等——都是在这篇论文中被确立的；尽管它最终模型的评估任务与能力，和今天人们所做的不一样。
自此，RLHF 被应用于多种任务。
重要的例子包括通用摘要 [@stiennon2020learning]、书籍的递归摘要 [@wu2021recursively]、指令跟随（InstructGPT）[@ouyang2022training]、浏览器辅助问答（WebGPT）[@nakano2021webgpt]、用引文支撑答案（GopherCite）[@menick2022teaching]，以及通用对话（Sparrow）[@glaese2022improving]。

除了应用之外，还有若干奠基性论文为 RLHF 的未来划定了关键领域，包括：

1. 奖励模型的过度优化 [@gao2023scaling]：RL 优化器对“以偏好数据训练出的模型”过拟合的能力；
2. 把语言模型作为对齐研究的一个通用领域 [@askell2021general]；
3. 红队测试 [@ganguli2022red]——评估语言模型安全性的过程。

针对聊天模型应用的 RLHF 精化工作持续推进。
Anthropic 在早期版本的 Claude 上大量使用它 [@bai2022training]，早期 RLHF 开源工具也随之出现 [@ramamurthy2022reinforcement]、[@havrilla-etal-2023-trlx]、[@vonwerra2022trl]。

## 2023 至今：ChatGPT 时代

ChatGPT 的发布公告非常明确地说明了 RLHF 在其训练中的作用 [@openai2022chatgpt]：

> 我们使用人类反馈强化学习（RLHF）训练了这个模型，方法与 InstructGPT 相同，但在数据收集设置上有细微差异。

自此以后，RLHF 被广泛用于领先的语言模型。
众所周知，它被用于 Anthropic 为 Claude 设计的宪法式 AI [@bai2022constitutional]、Meta 的 Llama 2 [@touvron2023llama] 与 Llama 3 [@dubey2024llama]、NVIDIA 的 Nemotron [@adler2024nemotron]、Ai2 的 Tülu 3 [@lambert2024t]，以及更多模型。

今天，RLHF 正在成长为更广的**偏好微调**（PreFT）领域，其中包括一些新应用：针对中间推理步骤的过程奖励 [@lightman2023let]（见第 5 章）；受直接偏好优化（DPO）启发的直接对齐算法 [@rafailov2024direct]（见第 8 章）；从代码或数学的执行反馈中学习 [@kumar2024training]、[@singh2023beyond]，以及受 OpenAI o1 启发的其他在线推理方法 [@openai2024o1]（见第 7 章）。
