<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
  代码块中的对话示例保留英文原文：它们是 tokenizer 实际产出的 token 序列，
  翻译会使其失去示范意义。
-->
---
prev-chapter: "训练总览"
prev-url: "03-training-overview"
page-title: 指令微调
search-title: "第 4 章：指令微调"
meta-description: "指令微调如何把基座语言模型变成可用的助手，并为后续的 RLHF 与后训练阶段做好准备。"
next-chapter: "奖励建模"
next-url: "05-reward-models"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "第 2 讲：IFT、奖励建模与拒绝采样（第 4、5、9 章）"
---

# 指令微调

早期的大型预训练语言模型以“下一词预测”为目标训练，默认并不带有一个明确的“遵循指令”接口。
在 GPT-3 发布前后 [@brown2020language]，提示（prompting）与上下文学习（in-context learning）成为让单个模型适配多种任务的常用方式（尽管针对具体任务做微调仍然普遍）——做法是在上下文中给出示例，再让模型完成类似的任务。
一个很自然的下一步就是**指令微调**：教模型以“指令—回答”的格式作答，而不只是续写文本。
例如，给定提示 “What is the capital of France?”，基座模型可能会续写成 “What is the capital of Germany? What is the capital of Italy?...”——只是把提问的模式延续下去；而一个经过指令微调的模型会回答 “The capital of France is Paris.”

指令微调在两股研究潮流的交汇处腾飞起来。
第一，NLP 从各自定制的微调任务设定，转向统一的“文本到文本”或指令式框架，这样一来，把五花八门的数据集标准化、并用一个模型训练多种任务就变得直接可行。
把任务框架统一起来的代表性工作包括 *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*（T5 系列）[@raffel2020exploring]、*Finetuned Language Models Are Zero-Shot Learners*（FLAN 数据集）[@wei2021finetuned]、*Multitask Prompted Training Enables Zero-Shot Task Generalization*（T0 系列）[@sanh2021multitask]，以及 *Cross-Task Generalization via Natural Language Crowdsourcing Instructions*（Natural Instructions 数据集）[@mishra2021cross]。
第二，预训练语言模型的规模化、以及提示/上下文学习的兴起表明：单个模型能够跨任务泛化；而当模型被明确地在“指令—回答”示例上训练过之后，这种泛化会可靠得多。
这两股潮流合在一起，开启了一个“在大量指令数据上微调预训练语言模型”的时代——也就是如今通常所说的**指令微调**（IFT）或**监督微调**（SFT），它让训练通用模型这件事对更广泛的群体变得可行。

自被发现以来，指令微调（口语上也常直接叫 *instruction tuning*）已经成熟，并成为许多语言模型流水线中的标准做法。
从核心上说，IFT 是让语言模型适配到某个目标任务分布的最简单方法。
它为 RLHF 打下基础：把模型准备好去接受一种称为“问答”的指令格式；同时，它也是人们把现代技术应用到新领域时所使用的第一个工具。
没有最基本的指令跟随能力，本书讨论的多数流水线——从偏好数据收集到在线 RLHF 优化——都无法进行。

指令微调本身在别处已有大量论述，并且它在核心上属于监督学习；所以本章聚焦于对 RLHF 实践者最重要的那些实操细节：**训练数据如何格式化与组织**。
关于数据和格式化的决策，会直接被后续训练阶段沿用，从而形成一套统一的“语言”，供模型吸收后训练数据。

## 聊天模板与指令结构

后训练流程的第一步，是定义一个模式来格式化用户查询，使那些通过 tokenizer 处理信息的语言模型能够轻松读懂。
当使用预训练语言模型时，提示非常简单。模型只认识少数几个 token：序列起始 token（如 `<bos_token>`）、序列结束 token（如 `<eos_token>`），以及填充 token（用于处理批次中存在空成分时的训练）。
这意味着，要给基座模型下提示，用户输入一段 token 序列让模型接着往下写，例如：

```text
<bos_token> The capital of the United States is
```

然后模型会持续生成 token，直到用尽上下文窗口、或者生成了序列结束 token。

所有后训练阶段——从指令微调到 RLHF 及其他方法——都依赖这套格式化来训练模型。
负责处理“与用户交互之结构”的工具，就叫**聊天模板**（chat template）。

下面是一个我们即将拆解的示例：

```jinja
{% if messages[0]['role'] == 'system' %}
    {# If the conversation begins with a system message, treat it as a special first turn.
       We set an offset so the user/assistant alternation check lines up correctly. #}
    {% set offset = 1 %}
{% else %}
    {# No system message: user should be the first non-empty turn. #}
    {% set offset = 0 %}
{% endif %}

{# Emit the beginning-of-sequence token (model-specific). #}
{{ bos_token }}

{# Serialize each message into the model's chat-markup tokens. #}
{% for message in messages %}
    {# Enforce role alternation: (system), user, assistant, user, assistant, ...
       The boolean expression compares "is this a user message?" against whether the
       current index (plus offset) is expected to be user or assistant. #}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {# Wrap each message with special tokens:
       - <|im_start|><role>\n
       - message content (trimmed)
       - <|im_end|>\n
       This produces a single flat token sequence the LM can train on. #}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{# Optionally append an "assistant" start tag with no content.
   This cues generation to continue from the assistant role. #}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```

这段原始代码做的事情是：把 Python 中一个由消息与角色构成的字典列表，转换成语言模型可以据此预测的 token。

传入模型的全部信息都被赋予一个**角色**（role）。
传统的三种角色是 `system`、`user` 和 `assistant`。

`system` 标签只用于对话的第一条消息；它承载给这个智能体的指令，这些文本既不会来自用户，也不会展示给用户。
这些**系统提示**（system prompt）用于给模型提供额外上下文（比如日期和时间），或者用来修补某些行为。
举个有趣的例子：可以告诉模型“你是一个友好的聊天机器人，总是用海盗的口吻回答”。

另外两个角色则很直接：**user** 承载使用 AI 的那个人发出的消息，**assistant** 承载模型给出的回复（也就是以 AI 助手身份参与对话）。

为了把这一切转换成 token，我们使用上面那段代码。模型有一系列*特殊 token*，用来把各条消息彼此分隔开。
如果用示例查询 “How many helicopters can a human eat in one sitting?” 运行上面的代码，传入模型的 token 序列会是这样：

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

注意序列最后的 token 是 `<|im_start|>assistant`。模型正是靠这个知道该继续生成 token，直到最终生成它的序列结束 token——在这个例子里就是 `<|im_end|>`。

把所有问答对数据（以及下游的偏好微调数据）都打包成这种格式之后，现代语言模型会以完全一致的方式遵循它。这就是指令微调模型用来与用户、以及与在 GPU 或其他计算设备上运行的模型交换信息的语言。

这种形式可以朴素地扩展到多轮对话，如下所示：

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
Oh just 6.<|im_end|>
<|im_start|>user
Are you sure about that?<|im_end|>
<|im_start|>assistant
```

在开源生态中，把聊天模板应用到消息列表上的标准做法，是使用一段存放在 tokenizer 配置里的 Jinja 片段，即 `apply_chat_template`。

上面这个聊天模板是 OpenAI 的 Chat Markup Language（ChatML）的衍生物——ChatML 是早期对消息格式做标准化的一次尝试。
如今，OpenAI 及其他模型提供方使用的是一套分层的体系：用户可以配置系统消息，但还存在更高层级的指令，它们可能披露、也可能不披露给用户 [@wallace2024instruction]。

此外还有许多其他的聊天模板。其他例子包括 Zephyr 的 [@tunstall2023zephyr]：

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

以及 Tülu 的：

```text
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
```

除此之外，许多聊天模板还会加入用于工具调用等任务的格式化内容与其他 token。


## 指令微调的最佳实践

把指令微调作为后训练、以及打造有帮助的语言模型的基础，这一点已经非常确定。
实现成功指令微调的方式有很多。
例如，对部分模型参数做量化以实现高效微调，能让训练变得非常容易上手 [@dettmers2023qlora]。
此外，在聊天对齐这类窄领域内（即不涉及数学、代码等更难技能），小而专注的数据集也能取得很强的表现 [@zhou2023lima]。

ChatGPT 发布后不久，像 No Robots 这样只有约 1 万条样本的人类数据集便是当时的最高水平 [@no_robots]。
几年之后，大规模合成数据集在大多数任务上表现最好 [@lambert2024t]。

有几条原则依然成立：

- **高质量数据是性能的关键。** 模型真正学习的是那些**补全**（在很多情况下，提示并不参与预测，所以模型并不学习预测提示）。
- 约 100 万条提示就足以训练出一个能很好地进行 RLHF 与后训练的模型。继续扩规模仍有帮助，但收益会迅速递减。
- 最好的提示，是与下游关注任务分布相近的那些。
- 如果指令微调之后还要做多个训练阶段，模型可以从指令微调数据中的一些噪声里恢复过来。**把整体优化设计好，比死抠每一个单独阶段更重要。**

## 实现细节

虽然损失函数与预训练所用的相同，但有一些关键的实现细节与预训练设定不同。
许多做法——比如决定用哪些并行方式来把模型切分到多张 GPU 上——与预训练相同，但使用的机器总数通常更少（对应下面列出的第一项技术差异）：

- **更小的批大小**：与预训练相比，指令微调（以及偏好微调等其他后训练技术）使用小得多的批大小，以便在更窄的数据分布上优化得当，同时保住模型从预训练获得的泛化能力。例如，OLMo 2 在 7B 和 13B 的预训练中分别使用 1024 和 2048 的打包行数批大小，这些模型的总上下文长度是 4096 个 token，批次中的每一行都是若干文档拼接、填满序列长度的结果。而在后训练中，这两个模型都只使用 256 个**提示**的批大小 [@olmo20242]，且不填充到完整序列长度（因此每批的非掩码 token 少得多）。更小的批大小意味着这些训练任务无法像预训练那样切分到那么多设备上——实践中，分布式训练设置有“每设备最小批大小”的限制，所以如果你想为 SFT 保留一个较小的全局批大小，就得相应地少用一些 GPU。而在实践中，批大小导致单个训练任务只能用更少的 GPU，并不构成瓶颈：因为 SFT 的训练 token 总量远小于预训练，而且后训练往往需要跑多个随机种子才能选出最好的最终性能。
- **提示掩码**：预训练时，批次中每个 token 都参与自回归预测，损失作用在所有 token 上。而在指令微调中，**提示 token 会被掩蔽掉**，这样模型就不会去学习准确预测用户的提问——只学预测回复。其他后训练算法同理。
- **多轮掩码**：对于多轮对话，有两种常见的掩码选择。(1) *只保留最后一轮*：只有最后一条 assistant 轮次中的 token 计入损失，之前的所有上下文（包括更早的 assistant 轮次）都被掩蔽。长对话仍可“展开”成多个训练样本：对一个 $N$ 轮的对话，每个样本预测一条 assistant 回复，同时掩蔽之前所有上下文、并排除之后的轮次。(2) *只掩蔽用户轮次*：所有 user 轮次都被掩蔽，但**每一条** assistant 轮次都计入损失。如果你想得到更多（更短的）训练样本，这种设定下同样可以展开；关键差别在于中间的 assistant 回复是直接参与训练的。
- **与预训练相同的损失函数：** 指令微调使用与预训练语言模型相同的自回归损失函数，但数据与掩码方式差别很大（只在完整序列上训练，而预训练的文档可以被切分到不同批次）等等。
- **学习率：** SFT 通常使用比预训练小一到两个数量级的学习率，以更好地应对不同的优化动态（数据集更小、批更小、以及一个很强的预训练初始化，都更倾向于保守的更新）。例如，OLMo 2 预训练的峰值学习率是 $3 \times 10^{-4}$，而 SFT 是 $1 \times 10^{-5}$ [@olmo20242]。Olmo 3 使用了更高的 SFT 学习率 $5\text{-}8 \times 10^{-5}$ [@teamolmo2025olmo3]，部分原因是它的训练基础设施使用了序列打包——把多个样本装进每条训练序列，从而提高了以“有效 token”衡量的批大小。更大的批产生方差更低的梯度估计，这又支持在训练不失控的前提下使用更高的学习率——这个关系被称为线性缩放律。学习率通常会在训练步数的一小部分内先做预热，然后线性衰减。实践中，团队往往会扫多个学习率，并在留出评估套件上选出最好的检查点 [@teamolmo2025olmo3]。

## 建议的实验

配套代码仓库里有一个小型的 SFT 训练脚本，位于 `code/instruction_tuning/`。
它的用意是一个学习练习，让“从基座模型到助手”的转变变得具体可见。

1. **跑一遍经典的 SFT 示例，观察“基座→助手”的转变。**
   运行：

   ```bash
   cd code/
   uv run python -m instruction_tuning.train --config instruction_tuning/configs/sft_olmo2_1b.yaml
   ```

   它会在 `HuggingFaceH4/no_robots` 上训练 `allenai/OLMo-2-0425-1B`（基座版），并每 50 个优化器步打印固定提示池的生成结果。
   在第 0 步，基座模型会漫无边际地乱写、重复提示、并吐出格式错误的角色标记；几百步之后，同样的提示会产出简洁、并以 `<|endoftext|>` 结束的回答。
   这就是指令微调的自检点——损失函数与预训练相同，只是作用在一个“提示 token 被掩蔽”的聊天模板上。

2. **扫学习率。**
   复制 `sft_olmo2_1b.yaml`，固定其他一切，分别试 `1e-6`、`5e-6` 和 `5e-5` 的 `lr`。
   观察在哪个学习率上模型最早学会作答并干脆地停下，又在哪个学习率上开始过拟合并产出“模板形状”的废话。
   这就是上面那条“比预训练低一到两个数量级”指导的实操版本。
