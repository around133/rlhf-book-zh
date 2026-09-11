<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容与引用。
  代码块、JSON/XML 数据与英文提示词保留原文。
-->
---
prev-chapter: "合成数据与蒸馏"
prev-url: "12-synthetic-data"
page-title: 工具使用与函数调用
search-title: "第 13 章：工具使用与函数调用"
meta-description: "作为后训练目标的工具使用与函数调用，用以构建能力更强的语言模型产品与智能体。"
next-chapter: "过度优化"
next-url: "14-over-optimization"
lectures:
  - video: "https://www.youtube.com/watch?v=GMry2DzC304&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=17"
    label: "第 11 讲：工具使用、函数调用与通往智能体之路"
---

# 工具使用与函数调用

让语言模型使用工具，是扩展其能力的一种自然方式——尤其对于那些信息在外部工具中、要求高精确度的任务，或需要与复杂网络系统交互的智能体而言。
工具使用是语言模型需要被训练才能具备的技能，而 RLHF 以及本书介绍的所有其他方法都能对它做精化。
考虑一个用户问题，例如：

> **用户**：今天谁是美国总统？

一个没有工具的语言模型会因为预训练数据的知识截止而难以回答这个问题，但这条信息只需一次搜索查询就能拿到。
再看另一个例子：

> **用户**：把我 downloads 文件夹里所有的 arXiv 论文移到 ~/research/ 目录，文件名要能体现论文日期。

这是仅靠模型权重根本无法尝试的任务——工具的使用让语言模型能够处理范围广阔得多的任务。

在深入之前，有必要区分几个常被混用的相关术语：

- **工具使用**（tool use）：模型输出一个结构化请求（工具名与参数）；一个编排器执行该工具；结果被追加到上下文中；模型继续生成。
- **函数调用**（function calling）：工具使用的一种，其中参数必须符合一组函数的声明式 schema（通常是 JSON Schema），从而实现可靠的解析与校验。
- **代码执行**（code execution）：工具使用的特例，其中“工具”是一个代码解释器（例如 Python）；结果作为工具输出返回。

## 工具使用总览

AI 模型通过输出特殊 token 来触发某个端点，从而使用任何外部工具。
这些工具可以是高度专用的函数（例如返回某地天气的函数），也可以是代码解释器或搜索引擎——它们构成复杂行为的基本积木。
我们的第一个例子展示了语言模型需要更新的信息，来补充“权重固定在过往数据上”这一特性；但也存在代码执行这类工具，让语言模型绕开自己概率式、生成式的本质，返回精确答案。
考虑这个任务：打印圆周率到 50 位的近似值（而不是凭记忆背诵、冒幻觉的风险）。
带工具的语言模型可以这样做：

```text
<code>
from decimal import Decimal, getcontext
getcontext().prec = 60

def compute_pi():
    # Chudnovsky algorithm for computing pi
    C = 426880 * Decimal(10005).sqrt()
    K, M, X, L, S = 0, 1, 1, 13591409, Decimal(13591409)
    for i in range(1, 100):
        M = M * (K**3 - 16*K) // ((i)**3)
        K += 12
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
    return C / S

print(str(compute_pi())[:52])
</code>

<output>
3.14159265358979323846264338327950288419716939937510
</output>
```

本章概览工具使用在现代语言模型中的起源、其基本原理与格式化方式，以及在领先模型中用好工具所涉及的各种权衡。

“工具使用”这个术语的确切来源并不清楚，但这个想法的起源远早于 RLHF 普及的后 ChatGPT 世界。
2015 年前后就有早期尝试，构建早于现代语言模型的系统，例如神经程序解释器（NPI）[@reed2015neural]——“一种学习表示与执行程序的循环、可组合神经网络”。
随着语言模型流行起来，许多子领域都在通过与外部能力集成来提升性能。
为获取权重之外的信息，许多人使用检索增强生成 [@lewis2020retrieval] 或网页浏览 [@nakano2021webgpt]。
不久之后，另一些人开始探索把语言模型与程序 [@gao2023pal] 或工具 [@parisi2022talm] 集成。

随着领域成熟，这些模型除了底层语言建模能力的巨大改进之外，还获得了更复杂的能力。
例如 Toolformer 可以使用“一个计算器、一个问答系统、两个不同的搜索引擎、一个翻译系统，以及一个日历” [@schick2023toolformerlanguagemodelsteach]。
不久之后，Gorilla 被训练来使用 1645 个 API（来自 PyTorch Hub、TensorFlow Hub v2 与 Hugging Face），其评测集 APIBench 成了流行的 Berkeley 函数调用排行榜的基础 [@patil2023gorilla]。
自这些早期模型以来，所调用动作的多样性已大幅增长。

工具使用模型如今与日常的语言模型交互深度交织。
模型上下文协议（Model Context Protocol，MCP）作为一种通用格式出现，用于把语言模型连接到外部数据源（或工具）[@anthropic_mcp_2024]。
有了更强的模型与更好的格式，工具使用型语言模型被用在许多场景中，包括 Microsoft Office 或 Google Workspace 这类流行应用中的生产力副驾驶、科学领域 [@bran2023chemcrow]、医学领域 [@li2024mmedagent]、编程智能体 [@zhang2024codeagent]（如 Claude Code 或 Cursor）、与数据库的集成，以及许多其他自主工作流。

评估工具使用模型涉及多个维度：工具名与参数正确性的精确匹配指标、schema 有效性，以及在模拟环境中的端到端任务完成度。
多次试验中的可靠性也很重要——$\tau$-bench 提出了 pass^k 指标（区别于 pass@k）来衡量一个智能体是**稳定成功还是偶尔成功** [@yao2024taubench]。
ToolLLM 及其 ToolBench 数据集提供了一个大规模框架，用于在 16000 多个真实 API 上训练与评估工具使用 [@qin2023toollm]；而 Berkeley 函数调用排行榜（BFCL）仍是比较模型函数调用准确性的流行基准 [@patil2023gorilla]。

## 在生成中交织工具调用

函数调用的训练数据看起来很像其他后训练数据，只是多了一样东西：一个告诉模型它有哪些工具可用的系统提示。
下面是一个格式化后的数据点示例，包含系统提示与以 JSON 格式给出的可用工具：
```xml
<system>
You are a function-calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
</system>

<functions>
[
  {
    "name": "search_movies",
    "description": "Search for movies by title and return matching results with IDs.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search string for the movie title."
        }
      },
      "required": ["query"]
    }
  },
  {
    "name": "get_movie_details",
    "description": "Fetch detailed information about a movie including cast, runtime, and synopsis.",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "The unique identifier for the movie."
        }
      },
      "required": ["movie_id"]
    }
  },
  {
    "name": "get_showtimes",
    "description": "Get movie showtimes for a given location and date.",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "The unique identifier for the movie."
        },
        "zip_code": {
          "type": "string",
          "description": "ZIP code for theater location."
        },
        "date": {
          "type": "string",
          "description": "Date for showtimes in YYYY-MM-DD format."
        }
      },
      "required": ["movie_id", "zip_code"]
    }
  }
]
</functions>

<user>
...
</user>
```
尽管语言模型是在生成补全，但如果它遵循这个示例，就会生成 token `search_movies("Star Wars")` 来搜索《星球大战》。
这通常被编码在特殊的格式化 token 里，随后插入序列的下一批 token 会包含工具的输出。
有了这些，模型就能学会完成比许多简单独立模型更困难的任务。

一种流行的工具使用形式是**代码执行**，它让模型能对复杂逻辑或数学问题给出精确答案。
例如，语言模型内的代码执行可以发生在推理模型的思考 token 里。
与函数调用一样，先有一个放待执行代码（由模型生成）的标签，再有一个单独放输出的标签。
```text
<|user|>
What is the 50th Fibonacci number? (Use the standard F_0=0, F_1=1 indexing.)</s>
<|assistant|>
<think>
Okay, I will compute the 50th Fibonacci number with a simple loop, then return the result.

<code>
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fib(50)
</code>

<output>
12586269025
</output>
</think>
<answer>
The 50th Fibonacci number is 12 586 269 025.
</answer>
```

底层发生的是：语言模型把工具的输入输出，与标准的自回归生成 token 交织在一起。
实现这件事的编排循环大致如下：

```python
messages = [...]
while True:
    response = model(messages, tools=tools)
    if not response.tool_calls:
        return response.text

    for call in response.tool_calls:
        result = execute_tool(call.name, call.args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

![工具使用把模型生成与外部执行交织起来：模型持续生成 token，直到它发出一个工具调用（橙色）；外部系统执行该工具并把输出（紫色）注入序列；随后模型继续生成。模型可以在单次生成中发出多个工具调用。训练时，工具调用与工具输出的 token 通常都会从损失中掩蔽。](images/tool_use_generation.png){#fig:tool-use-generation data-dark-src=“images/tool_use_generation-dark.png”}

工具使用的训练，就是让模型在这种不同的 token 流下表现得可预测——知道何时该发出工具调用、如何正确格式化参数、以及如何把结果整合进回答。
开放模型必须被训练成能与用户可能即插即用的各种工具协同工作。

## 多步工具推理

OpenAI 的 o3 模型代表了多步工具使用与语言模型集成方式的一次重大跃迁。
这种行为与社区中早得多的研究趋势相关。
例如 ReAct [@yao2023react] 展示了如何把动作与推理交织进同一次模型生成：

> 在这篇论文中，我们探索让 LLM 以交织的方式同时生成推理轨迹与任务特定动作，从而使两者产生更大的协同：推理轨迹帮助模型归纳、跟踪并更新行动计划，也能处理异常；而动作让它能与知识库或环境这类外部信息源对接，并从中收集额外信息。

随着工具使用能力的固化与推理模型的起飞，**多轮工具使用**已成长为一个激动人心的研究领域 [@wang2025ragenunderstandingselfevolutionllm]。
用 RL 训练这些多步行为，比逐样本的 RLHF 循环更接近经典强化学习：智能体在一个完整轨迹上与环境和工具交互，之后才被赋予奖励，如 @fig:tool-use-rl 所示。

![多步工具使用的强化学习。从训练数据采样一个提示，智能体（策略 $\pi_\theta$）在一条轨迹上与环境和工具交互，动作 $a_t$ 与观测 $o_t$ 交替出现。完成的轨迹被打分或验证，在末尾产生一个单一奖励 $r_T$，驱动策略更新。与逐样本的 RLHF 循环不同，奖励只在多步 rollout 之后才到来——更接近经典 RL。](images/tool_use_rl_loop.png){#fig:tool-use-rl data-dark-src=“images/tool_use_rl_loop-dark.png”}

## 模型上下文协议

模型上下文协议（Model Context Protocol，MCP）是一个开放标准，用于把语言模型连接到外部数据源与信息系统 [@anthropic_mcp_2024]。
在数据层，MCP 使用 JSON-RPC 2.0，其原语带有发现（discovery）与执行（execution）方法。
它不要求为每个外部系统规定特定的工具调用格式，而是让模型通过一个标准化协议访问丰富的上下文信息。

MCP 是在本章工具使用内容之上的一个简单补充——它是应用以可预测的 JSON schema 把上下文（数据 + 动作）传给语言模型的方式。
模型与之交互的 MCP 服务器有核心原语：资源（只读数据块）、提示（模板化消息/工作流）和工具（模型可调用的函数）。
由此，MCP 架构可以概括为：

- MCP 服务器包装某个特定的数据源或能力。
- MCP 客户端（例如 Claude Desktop、IDE 插件）聚合一个或多个服务器。
- 宿主（例如 Claude 或 ChatGPT 应用）提供用户/LLM 界面；更换模型供应商或后端工具，只需替换中间那一层客户端。

MCP 让工具使用模型的开发者能用同一套基础设施，把自己的服务器或客户端接到不同模型上；同时模型也有了一个可预测的格式去集成外部组件。
两者合在一起，为真实场景中的工具使用模型带来了可预测得多的发展环境。

一个 MCP 服务器通过标准化的 JSON schema 向客户端暴露工具：
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      }
    },
    "required": ["location"]
  }
}
```

一个实现该工具的最小 Python MCP 服务器：
```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="get_weather",
        description="Get current weather",
        inputSchema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        weather = fetch_weather(arguments["location"])
        return [TextContent(type="text", text=weather)]
```

## 实现细节

实现一个工具使用模型时，有多种格式化与掩码决策：

- **Python 还是 JSON 格式**：本章给出的例子中，工具使用既有格式化为 JSON 数据结构的，也有格式化为 Python 代码的。模型往往只选定一种结构，而业界不同提供方使用的格式各不相同。
- **掩蔽工具输出**：训练工具使用模型时的一个重要细节，是**工具输出中的 token 会从训练损失中掩蔽**。这确保模型不会去学习预测“处理工具调用之系统的输出”（因为那些结果并非模型生成的 token）。
- **工具调用的多轮格式化**：实现工具调用模型时，常见做法是给数据加载格式增加更多结构。后训练数据集的标准做法是一个在用户与助手之间交替的消息列表（通常还有一条系统消息）。工具使用的整体结构相同，但模型的轮次被按每次工具调用切分成内容子段。示例如下。

```python
messages = [
{
"content": "You are a function calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.",
"function_calls": null,
"functions": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Retrieve live giveaways from the GamerPower API based on the specified type.\", \"parameters\": {\"type\": {\"description\": \"The type of giveaways to retrieve (e.g., game, loot, beta).\", \"type\": \"str\", \"default\": \"game\"}}}]",
"role": "system"
},
{
"content": "Where can I find live giveaways for beta access and games?",
"function_calls": null,
"functions": null,
"role": "user"
},
{
"content": null,
"function_calls": "live_giveaways_by_type(type='beta')\nlive_giveaways_by_type(type='game')",
"functions": null,
"role": "assistant"
}
]
```

- **分词与消息格式细节**：OpenAI 消息格式中的工具调用，通常会经聊天模板（控制发给模型的消息格式的代码）做分词，把结构化的 JSON 表示转换成原始 token 流。这个过程因模型架构而异——有些用特殊 token 来界定工具调用，另一些则在 token 流内部保留结构化格式。[聊天模板演练场](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-8B) 提供了一个交互环境，可以探索不同模型如何把消息格式转成 token 流。
- **推理 token 的连续性**：随着推理模型出现——它们在答案之前有一段独立的“推理”token 流——对于在工具使用循环中如何处理它们，存在不同实现。有些模型在单轮内的多个工具调用步骤之间**保留**推理 token，从而在多次工具调用间维持上下文。但这些 token 通常会在轮次之间被清除以降低服务成本（不过并非总是如此——这是一个设计决策）。
- **各提供方的 API 格式**（截至 2026 年 5 月）：不同提供方使用概念相似但技术上有别的格式。OpenAI 的 Chat Completions API 使用带唯一 ID 的 `tool_calls` 数组，而更新的 Responses API 把调用表示为 `function_call` 条目，并以 `call_id` 为键把结果作为 `function_call_output` 条目返回。Anthropic 用 `input_schema` 定义工具，把调用与结果表示为 `tool_use` 与 `tool_result` 内容块。Gemini 暴露了 `AUTO`、`ANY`、`NONE` 等函数调用模式，并在受支持的 Gemini 与 Vertex AI 配置中提供 `VALIDATED`。
- **Schema 一致性与受限解码**：生产系统常用受限解码或“严格模式”选项来强制生成合法 JSON 与正确的参数类型，从而减少因格式错误导致的重试。一些闭源模型提供方会额外做后训练，专门让结构化 JSON 输出变得可靠；而对开放模型，这在 vLLM 这类系统中作为一个推理开关来处理。
- **工具输出对上下文的消耗**：工具输出可能很快吃光模型的上下文窗口，尤其当搜索或检索工具返回大量结果时。系统必须决定如何截断、摘要或分页工具输出，以便在保留模型继续所需信息的同时让上下文可控。

把这些联系回后训练：工具使用的训练数据从哪来、用什么目标？
人工撰写的工具轨迹采集成本很高，所以现代工具使用语料大多是合成的或自举的——Toolformer 式的自标注 [@schick2023toolformerlanguagemodelsteach]，或像 ToolBench 那样的大规模生成 [@qin2023toollm]。
训练目标方面：在工具轨迹上做监督微调（SFT）教会基本格式与工具选择。
这为行为提供了自举，通常就足以打下这项技能的基础。
在轨迹上做偏好优化（例如 DPO）可以改进“何时该调用工具、何时直接回答”的决策。
对于多步工具使用的智能体任务，带环境反馈（任务成功、约束满足）的 RL 就成了自然的目标——模型从“它的工具增强动作是否真的解决了问题”中学习。
