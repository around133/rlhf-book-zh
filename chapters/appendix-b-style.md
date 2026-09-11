<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容与引用。
  两段模型补全示例保留英文原文（英文版正是通过它们展示风格差异）。
-->
---
prev-chapter: "定义"
prev-url: "appendix-a-definitions"
page-title: "附录 B：超越「只是风格」"
search-title: "附录 B：超越「只是风格」"
meta-description: "RLHF 与后训练在语言模型中塑造的风格、校准与信息呈现问题。"
next-chapter: "实践问题"
next-url: "appendix-c-practical"
lectures:
  - video: "https://www.youtube.com/watch?v=y04JhXpiI4s&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y"
    label: "第 9 讲：过度优化与 RLHF 的坏名声"
---

# 超越「只是风格」

RLHF 早期的发展给它留下了一个名声：“不过是风格迁移”；另一些严厉的批评则聚焦 RLHF 如何操纵输出中信息的呈现方式。
本附录解释为什么**风格**是理解 RLHF 所提供价值的核心——以及它为何同时正面影响模型能力与用户体验。

“RLHF 只关乎风格迁移”这个说法之所以拖累了 RLHF 的叙事，有两个原因。
第一，它把 RLHF 与“对模型微小、不重要的改动”联系在一起。
人们谈论风格迁移时，往往不认为它重要或令人兴奋——他们觉得它肤浅。
然而，**风格是人类价值的无尽源泉**：它正是为什么重讲一个故事能产生新的畅销书（例如 [Sapiens](https://en.wikipedia.org/wiki/Sapiens:_A_Brief_History_of_Humankind)），也是我们智力生态持续进步的根本组成部分。
风格与“信息是什么”是交织在一起的。

第二个原因，是许多人忽略了：**做得好的 RLHF 会提升在流行 LLM 评估上的分数**。
我们已经看到，不同风格确实能实质性地改善 Llama 3 的评测表现 [@dubey2024llama]。
Llama 3 Instruct 系列在 Arena 上得分极高，人们普遍认为这是因为它们有更讨喜的个性——比同代其他模型更简洁、更机灵。
撇开许多 LLM 用户执迷的基准分数不论，如果 RLHF 能让语言模型单纯变得**更有趣**，那本身就是交付了价值。

本附录通篇用“话痨感”（chattiness）这个词来涵盖 RLHF 训练模型回复长度的增长，也涵盖大量使用 Markdown、表情符号、以及用项目符号列表组织答案等做法。
为让本附录的讨论落到实处，下面给出同一 Tülu 3 家族两个模型对同一提示的示例补全——一个只做了指令微调，另一个通过长度受控的 DPO 做了偏好微调。
先看指令微调版本对提示 `What is reinforcement learning from human feedback?` 的补全（具体模型是 [Llama 3.1 Tulu 3 70B SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT)）[@lambert2024t]：

```text
Reinforcement learning from human feedback, also known as human-in-the-loop reinforcement learning, is a subfield of machine learning where the primary goal is to train reinforcement learning agents to perform tasks by incorporating feedback from humans. This feedback can come in various forms, such as binary rewards (e.g., right or wrong), scalar rewards, preferences between actions, demonstrations, natural language instructions, or corrections to the agent's behavior.

The idea behind reinforcement learning from human feedback is to leverage human intuition and decision-making abilities to guide the learning process of the agent. In traditional reinforcement learning, an agent learns to make decisions by interacting with an environment and receiving rewards or punishments based on the outcomes of those interactions. However, defining a suitable reward function can be challenging and time-consuming, often leading to reward shaping or reward hacking, where the agent finds unexpected and often unintended ways to maximize the reward.

By using human feedback, reinforcement learning can address some of these issues by allowing humans to directly influence the learning process. Human trainers can provide real-time feedback, scale their feedback to match the complexity of the task, and update their feedback based on the agent's progress. This approach can lead to more robust and generalized agents, as they are guided not by handcrafted reward functions but by the nuanced understanding and experience of human trainers.

However, reinforcement learning from human feedback faces several challenges, such as the high cost of obtaining human feedback, subjective and inconsistent feedback, scaling issues, and methods for efficient and effective integration of human feedback into learning algorithms. Despite these challenges, the research in this area has shown promising results in various applications, including robotics, video games, prosthetics control, natural language processing, and many more.
```

再看 RLHF 之后的补全，来自 [Llama 3.1 Tulu 3 70B DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO)：

```text
Reinforcement learning from human feedback (RLHF) is a subset of reinforcement learning (RL) where the rewards that guide the learning process are provided by humans, rather than a pre-specified numerical reward function. This is particularly useful in situations where it is hard or infeasible to design a suitable reward function, or where the desired performance involves subjective judgments that humans are uniquely equipped to provide.

In traditional reinforcement learning, an agent interacts with an environment. With each interaction, the agent performs an action, receives an observation, and gets a numerical reward that tells it how well it performed in that situation. Through many such interactions, the agent learns to maximize the cumulative reward.

In contrast, in reinforcement learning from human feedback:

1. **Human Input:** Instead of a predefined reward function, the feedback comes from human evaluators, who provide qualitative, subjective judgments about the agent's performance. This feedback can be in various forms, such as numerical ratings, binary preference signals, natural language feedback, or even demonstrations showing how to improve.

2. **Feedback Loop:** Human evaluators observe the agent's actions or outcomes and provide feedback in real-time (or close to real-time). The agent learns through this iterative feedback loop, gradually adapting its behavior to align more closely with human preferences.

3. **Reward Shaping:** Because raw human feedback is not always directly usable in learning algorithms due to variability and potential bias, a reward shaping process may be involved, whereby human feedback is converted into a more stable signal that guides learning.

4. **Preference-based RL:** One common approach within RLHF is preference-based reinforcement learning, where the agent is trained by comparing pairs of actions (or outcomes) and receiving human feedback on which one is better. From these pairwise comparisons, the agent infers a reward function that it can use to guide its actions.

5. **Inverse Reinforcement Learning (IRL):** Another technique used in RLHF is IRL, which aims to infer the underlying reward function that could explain human actions or decisions, and then use that reward function to train the agent.

**Applications:**

Reinforcement Learning from human feedback has found applications in:

- **Robotics:** Teaching robots to perform tasks that are difficult to specify with predefined reward functions.
- **Game playing:** Human feedback can guide AI agents to better emulate human-like strategies or behaviors.
- **Language and content generation:** For example, suggestively guiding AI models to generate more coherent, creative, or helpful content.
- **Recommendation systems:** Human feedback can help adjust and refine the recommendations provided to users.

**Challenges:**

Using human feedback introduces some specific challenges:

1. **Scalability:** Human feedback is time-consuming and expensive to collect, and it's hard to scale to complex or high-dimensional tasks.
2. **Bias and Noise:** Human feedback can be noisy, inconsistent, and biased, leading to suboptimal learning outcomes.
3. **Alignment:** Ensuring that the reinforcement learning agent aligns with the true preferences of the human evaluator(s) and not just the superficial feedback is difficult.
4. **Safety and Misuse:** There can be concerns about the safety and ethical implications of using potentially flawed human feedback, especially if the feedback process is corrupted or gamed.

Researchers are actively working on techniques to mitigate these challenges and improve RLHF approaches. One such approach is leveraging machine learning models to amplify the efficacy of the human feedback, for example, by learning to generalize from sparse human input or by synthesizing plausible feedback in situations where direct human input is not available.
```

DPO 模型的回答所含信息与前者相似（两个模型都基于同一个预训练基座），但偏好微调带来了更多结构、排版，当然还有更长的篇幅——而篇幅反复被证明与语言模型上“人类偏好”的一般观念相关。
更多此类例子见本书配套的补全库：[https://www.rlhfbook.com/library](https://rlhfbook.com/library)。

## 话痨感的平衡

无数人用 RLHF 或偏好微调方法来提升 AlpacaEval 及其他自动聊天排行榜的分数（这些榜用“LLM 作为评判者”来近似衡量一个智能体在简单对话任务上有多有帮助、无害、诚实），但 RLHF 在这里带来的巨大增益，并**没有**同等比例地迁移到 Arena 这类更难钻空子的评估上。
张力在于：RLHF 方法确实在这些模型上给出可测量的改进，但这种训练并不总能转化为人们真正在乎的性能。
随着 RLHF 文献的建立，一大批模型带着相关方法发布，声称提升了模型的“对齐”，但它们往往做过了头，公布的评测分数从“有误导性”到“毫无意义”都有。

这些出于对齐动机的 RLHF 方法，若做对了，会让模型更好用、更愉快。
这通常伴随 MT-Bench 或 AlpacaEval 这类评估工具的明确提升。

2023 年秋，围绕直接偏好优化（DPO）相对于近端策略优化（PPO）及其他基于 RL 的偏好微调方法的角色，争论达到顶峰——“聊天评估分数与真实世界性能之间的平衡”正是争论的核心（关于这些权衡的更多技术讨论，见第 8 章、Ivison 等 2024 [@ivison2024unpacking]，或[这场演讲](https://youtu.be/YJMCSVLRUNs)）。
问题在于：你也可以用 DPO、PPO 这类技术配合反馈回路或海量数据，以换取这份聊天表现——而代价是**严重损害**模型在数学或代码等其他任务上的能力。

在 DPO 与 PPO 之争蔓延期间，许多论文给出了惊人的基准成绩，却没有获得实质采用。
即便这些论文发布了模型权重，它们在公开使用中也不受欢迎，因为这些模型在通用场景下并不鲁棒。
在 2023 年秋或稍后应用 RLHF 时，**没有任何办法让一个“对齐过的”70 亿参数模型在综合基准上真正击败 GPT-4**（这个判断会一直成立：当时的小模型无法稳健地击败最强的大型前沿模型）。
这似乎显而易见，但总有人发表声称这类结果的论文。
@fig:DNO 出自一篇名为 Direct Nash Optimization（DNO）的论文，它主张自己的模型在 2024 年 4 月是 7B 模型里 AlpacaEval 上的最先进水平 [@rosset2024direct]。
作为背景：DNO 是“奖励模型 + PPO”（经典 RLHF）或一次性 DPO 的一种批量式、同策略*迭代*替代方案，它把对齐框架化为“针对偏好 oracle 求纳什均衡”，从而直接优化成对偏好（胜率差）。
当学术激励与“全社会极度关注的技术”相遇时，这类挑战就会出现。

![Direct Nash Optimization（DNO）论文的结果，突出其小模型超越了 GPT-4 之类。Rosset 等 2024。许可 CC-BY。](images/dno-figure.png){#fig:DNO width=550px}

即便是 2024 年 1 月那篇先驱性论文 Self Rewarding Language Models [@yuan2025selfrewardinglanguagemodels]，也在 Llama 2 70B 上公布了不切实际的高分。
当然，那时 70B 模型确实能比 7B 模型更接近 GPT-4（正如 2024 年令人印象深刻的 Llama 3 发布所示），但把“模型的真实水平”与“现代 RLHF 论文中的声称”区分开很重要。
这些模型是为窄测试集调出来的，在真实使用中面对它们声称击败的、大得多的模型时，表现撑不住。
类似的方法来来去去还有很多，它们分享了有价值的洞见，也分享了被夸大的结果——这让 RLHF 更难理解。

被套用“怪味 RLHF”的模型，其一个常见症状是**长度偏置**。
这变得如此普遍，以至于 AlpacaEval 与 WildBench 等多个评估系统都内置了线性长度校正机制。
这修补了“靠话痨 doping 去击败 GPT-4 或当日领先前沿模型”的激励，造出一个不那么容易钻空子的动态——更短、更有用的模型可以真正获胜。

无论如何，只为“话痨感”而对齐聊天模型，如今在文献中有了一点名声上的代价——人们承认这些窄方法会以其他方式损害模型。
下面这条来自 2023 年阿里巴巴初代 Qwen 模型的说明，在早期对齐实验中被反复观察到，夸大了一种“话痨感 vs 性能”的权衡 [@qwen]：

> 我们用大量数据预训练了模型，并用监督微调和直接偏好优化对模型做了后训练。然而，DPO 带来了人类偏好评估上的提升，却造成了基准评估上的退化。

把这种权衡处理得当的一个早期好例子，是 2024 年 3 月的 Starling Beta 这样的模型 [@zhu2024starling]。
它从另一个聊天模型 OpenChat 微调而来 [@wang2023openchat]（实际上是由另一个完全不同的组织训练的）。
它的训练完全聚焦于 K 元奖励模型训练与 PPO 优化，在 Arena 上上升了 10 个名次。
模型的平均回复长度增加了，但增加的方式足以真正帮助人类评分者。
后来的例子如 Olmo 3 有记录显示经历了大量聊天训练，但作者偏好选择数学、代码与推理分数更高的最终检查点，而不是那些在基于 LLM 评判的聊天基准上最高的潜在检查点 [@teamolmo2025olmo3]。

一个自然的问题是：**为什么 RLHF 会让模型回复变长？**
根本上，Arena 这类评估告诉我们：与简短的回复相比，模型的普通用户往往更喜欢更长、更完整的答案。
更长的答案对快速评估它们的用户来说，可能感觉更周全、更有帮助，甚至更可信。
这并不代表**每一位**用户的偏好，但这些模型是按众多数据标注者的**平均**偏好训练的，所以 RLHF 倾向于让模型更啰嗦。
