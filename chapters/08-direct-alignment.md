<!--
  原文版权 (c) 2025-2026 Nathan Lambert，依 CC BY-NC-SA 4.0 许可发布:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  完整许可: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS

  本文件为个人学习用途的中文翻译，未改动原文的技术内容、公式与引用。
  代码块与英文提示词保留原文。
-->
---
prev-chapter: "推理与推理时扩展"
prev-url: "07-reasoning"
page-title: 直接对齐算法
search-title: "第 8 章：直接对齐算法"
meta-description: "DPO 等直接对齐算法：不依赖显式奖励模型或 RL 循环，直接优化偏好目标。"
next-chapter: "拒绝采样"
next-url: "09-rejection-sampling"
lectures:
  - video: "https://www.youtube.com/watch?v=6g6b4gvO-y0&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=8"
    label: "第 6 讲：直接偏好优化"
  - video: "https://www.youtube.com/watch?v=rhA7pLVt4E0&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=16"
    label: "对话 2：DPO 的实践"
---

# 直接对齐算法

直接对齐算法（Direct Alignment Algorithm，DAA）让人**无需训练中间的奖励模型、也无需使用强化学习优化器**，就能更新模型以求解同一个 RLHF 目标。
DAA 解决的是我们一直在研究的同一个偏好学习问题（用的是字面意义上相同的数据！），目的是让语言模型更对齐、更聪明、更好用。
没有奖励模型和在线优化，让 DAA 的实现简单得多，减少了训练期的算力开销，也让实验更容易。
本章详述推导这些算法所涉及的复杂数学，然后展示：那些有时颇为繁琐的推导，最终落到的实现却很简单。

最著名的 DAA、也是催化了整场语言模型对齐学术运动的算法，是**直接偏好优化**（Direct Preference Optimization，DPO）[@rafailov2024direct]。
核心上，DPO 用梯度上升求解同一个带约束的 RLHF 目标（见第 3 章）：

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:review_rlhf}

自 2023 年 5 月发布以来，在社区花了一段时间才摸清该用什么样的数据和超参之后（具体来说，是意外地低的学习率），许多流行模型都使用了 DPO 或其变体：从 2023 年 10 月以 Zephyr-$\beta$ 打头阵 [@tunstall2023zephyr]，到 Llama 3 Instruct [@dubey2024llama]、Tülu 2 [@ivison2023camels] 与 Tülu 3 [@lambert2024t]、Nemotron 4 340B [@adler2024nemotron] 等。
严格来说，序列似然校准（SLiC-HF）才是第一个发布的现代直接对齐算法 [@zhao2023slic]，但由于多种因素没能流行起来（要扭转一种研究方法被采用的惯性，从来都是棘手的事）。

DPO 与 DAA 影响最大的地方，是**降低了语言模型后训练的实验门槛**——它算力需求更低、更容易从零实现，也更容易在玩具和生产级示例上都跑通。

*本章通篇用 $x$ 表示提示、$y$ 表示补全。这一记号在语言模型文献中很常见——那些方法作用于完整的“提示—补全”对，而非单个 token。*

## 直接偏好优化

这里我们解释 DPO 如何工作的直觉，并完整重新推导其核心方程。

### DPO 如何工作

从表面看，DPO 就是直接优化一个策略来求解 RLHF 目标。
它的损失函数——我们稍后会在推导中重新审视——比较的是：学到的策略在被选中与被拒绝补全上的概率，相对参考模型发生了多大偏移。
由 Bradley-Terry 奖励模型推出的损失函数如下：

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_{\theta}(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right] $$ {#eq:dpo_core}

sigmoid 内部，第一项 $\beta \log \frac{\pi_{\theta}(y_c | x)}{\pi_{\text{ref}}(y_c | x)}$ 衡量策略相对参考模型把*被选中*补全的概率提高了多少，第二项对*被拒绝*补全做同样的度量。当被选中的提升超过被拒绝的提升时——即策略学会偏好正确回答时——损失下降。

通篇的 $\beta$ 是一个超参，用来在奖励优化与“最终模型同初始参考模型之间的 KL 散度”之间做平衡（也就是在平衡过度优化，这是正确使用 DPO 的关键超参）。
这依赖于 DPO 训练中**替代外部奖励模型**的隐式奖励——概率的对数比值：

$$r(x, y) = \beta  \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ {#eq:dpo_reward}

其中 $\pi_r(y \mid x)$ 是我们要求解的精确最优奖励策略。
它来自“把 Bradley-Terry 奖励用最优策略表达”（见 @eq:dpo_opt_policy），如第 5 章 Bradley-Terry 模型那一节所示。
本质上，正如 DPO 论文所说，这种重参数化给了我们“用最优策略而非奖励模型表达的、人类偏好数据的概率”——这意味着我们可以完全绕过显式的奖励模型学习。

来看 @eq:dpo_core 中优化器必须降低的那个损失。
当被选中回答的对数比值大于被拒绝回答的对数比值（都相对参考模型归一化）时，损失就更低。
实践中，这是模型在数据所呈现的 token 序列上对数概率的求和。
因此，DPO 做的是**拉大被选中与被拒绝回答之间的相对对数概率差距**。

有了 @eq:dpo_reward 中的奖励，我们可以写出损失的梯度，以进一步解读正在发生什么：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[ w \cdot \left(\nabla_{\theta}\log \pi_{\theta}(y_c \mid x) - \nabla_{\theta}\log \pi_{\theta}(y_r \mid x)\right) \right]$$ {#eq:dpo_gradient}

其中 $w = \sigma\!\left(r_{\theta}(x, y_r) - r_{\theta}(x, y_c)\right)$。

这个梯度通过以下方式求解上述目标：

- sigmoid 函数 $\sigma(\cdot)$ 内的第一项产生一个 0 到 1 的参数更新权重；当奖励估计**不正确**时它更大。当被拒绝样本反而比被选中样本更受偏好时，权重更新应该更大！
- 其次，内层方括号 $[\cdot]$ 中的项提高被选中回答 $y_c$ 的似然、降低被拒绝回答 $y_r$ 的似然。
- 这些项被 $\beta$ 加权，$\beta$ 控制更新的力度如何在“把补全顺序排对”与“KL 散度约束”之间平衡。


核心直觉是：**DPO 在拟合一个隐式奖励模型，而它所对应的最优策略可以闭式地提取出来**（@eq:dpo_opt_policy，这要归功于梯度下降和我们手头的 ML 工具）。
因为 DPO 损失可直接求导，算出精确梯度是直接的，不需要“训练奖励模型、采样补全打分”来估计它。
经常被误解的一点是：**DPO 在核心上是在学一个奖励模型**——这正是论文副标题《Your Language Model is Secretly a Reward Model》的由来。
人们很容易把它和“DPO 目标直接训练一个策略”混淆，所以研究下面的推导有助于获得完整的理解。

有了这个隐式奖励模型的学习，DPO 就能在给定数据集中的数据、以及目标中那个特定 KL 约束 $\beta$ 之下，产出 RLHF 目标的最优解。
这里，DPO 之所以能针对一个特定 KL 散度求出精确策略，是因为它的生成不是像策略梯度算法那样在线的——这是它与偏好微调 RL 方法的核心差别。
在很多方面，这让 DPO 的 $\beta$ 比在线 RL 方法更容易调；但关键在于、也符合直觉的是：**最优值取决于被训练的模型与训练它的数据**。

在每一批偏好数据（由许多补全对 $y_{chosen} \succ y_{rejected}$ 组成）上，DPO 直接朝最优解做梯度步。
它比策略梯度方法简单得多。

![DPO 刚发布时，在研究社区引发了一场关于“如何最好地做 RLHF 与偏好学习”的激烈争论。这张梗图很好地捕捉了当时的情绪——那场争论常常让人觉得是被逼出来的、用力过猛，但许多刚入门的人乃至顶级实验室都从 DPO 中获得了巨大收益。DPO 简洁性梗图，作者 Tom Goldstein。](images/dpo_meme.jpeg){#fig:dpo-meme}


### DPO 推导

DPO 的推导有两个主要部分。
第一，作者给出了**最优求解 RLHF 目标**的策略形式——也就是本书通篇使用的那个目标。
第二，他们展示了如何从成对偏好数据（即 Bradley-Terry 模型）抵达这个解。

#### 推导 RLHF 的最优解

首先，我们应当再看一眼 RLHF 优化目标——这里表明我们希望最大化这个量：

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:rlhf_opt_eq_repeat}

这里的双重期望只作用于“计算期望奖励所需的采样”，因为 KL 项本身仍是一个解析表达式。
首先，展开 KL 散度的定义。回忆 $\mathcal{D}_{\text{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$，其中求和里的 $\pi(y|x)$ 权重变成了采样分布。
由于两项现在共享同一个对 $y \sim \pi(y|x)$ 的期望，可以把它们合并：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] $$ {#eq:dpo_deriv_1}

接下来，把负号从括号中的差里提出来。为此把它拆成两项：

$$ = \max_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)\right] - \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_2}

然后乘 $-1$，把最大化转成最小化：

$$ = \min_{\pi}\left(-\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)\right] + \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_3}

除以 $\beta$ 并重新合并：

$$ = \min_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) \right]\right) $$ {#eq:dpo_deriv_4}


接下来，必须引入一个配分函数 $Z(x)$：

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

配分函数充当未归一化密度 $\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 的归一化因子，从而使它对每个固定的 $x$ 都成为 $y$ 上的合法概率函数。它的确切必要性很快会在推导中显现。

把它代入，我们得到中间变换：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right] $$ {#eq:dpo_deriv_5}

要看这是怎么来的，考虑 @eq:dpo_deriv_4 方括号内的优化内部部分：

$$ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_6}

然后加上 $\log Z(x) - \log Z(x)$：

$$ = \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) + \log Z(x) - \log Z(x) $$ {#eq:dpo_deriv_7}

然后重新分组：

$$ = \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x) \right) - \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_8}

利用 $\log(x) + \log(y) = \log(x\cdot y)$（并把 $Z$ 移到分母），得到：

$$ = \log \frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)}- \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_9}

接下来，把 $\frac{1}{\beta}r(x,y)$ 展开为 $\log \exp \frac{1}{\beta}r(x,y)$，做同样操作即可得到 @eq:dpo_deriv_5，这里稍微改写一下：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}} \left[ \mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} \right] - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

用这个优化形式，我们需要真正解出最优策略 $\pi^*$。
由于我们引入了配分函数 $Z(x)$，从而使 $\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 成为 $y$ 上的合法概率分布，我们可以辨认出：**内层期望实际上就是一个真正的 KL 散度**！

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathcal{D}_{\text{KL}} \left(\pi(y|x) \middle\| \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_11}

由于 $\log Z(x)$ 这一项不依赖 $\pi$（我们正在优化的策略），可以忽略它。剩下的就只有“我们正在学习的策略”与“一个把配分、$\beta$、奖励和参考策略联系起来的分布”之间的 KL 散度。
吉布斯不等式告诉我们：它只在两者相等时才取到 0 这一最小值！
于是我们得到最优策略：

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_opt_policy}


#### 为 BT 模型推导 DPO 目标

首先回忆：如第 5 章「奖励建模」与第 11 章「偏好数据」所述，人类偏好的 Bradley-Terry 模型形如：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right) + \exp\left(r^*(x, y_2)\right)} $$ {#eq:bradley_terry_dpo}

通过变换 @eq:dpo_opt_policy，我们可以解出最优奖励。先对两边取对数：

$$\log \pi^*(y|x) = \log \left( \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r^*(x,y)\right) \right)$$ {#eq:dpo_reward_deriv1}

用 $\log(abc) = \log a + \log b + \log c$ 展开右侧：

$$\log \pi^*(y|x) = -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta}r^*(x,y)$$ {#eq:dpo_reward_deriv2}

移项解出 $r^*(x,y)$：

$$\frac{1}{\beta}r^*(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)$$ {#eq:dpo_reward_deriv3}

两边乘 $\beta$：

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$ {#eq:dpo_reward_full}

然后把这个奖励代入 @eq:bradley_terry_dpo 的 Bradley-Terry 方程，得到：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} + \beta \log Z(x)\right)} $$ {#eq:dpo_loss_deriv0}

把指数表达式从 $e^{a+b}$ 拆成 $e^a e^b$，再约掉 $e^{\beta \log Z(x)}$ 这一项，它化简为：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right)} $$ {#eq:dpo_loss_deriv1}

接着，分子分母同乘 $\exp\left(-\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)$：

$$p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)} $$ {#eq:dpo_loss_deriv2}

最后，用 sigmoid 函数的定义 $\sigma(x) = \frac{1}{1+e^{-x}}$，得到：

$$p^*(y_1 \succ y_2 \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right) $$ {#eq:dpo_loss_deriv3}

在最优策略 $\pi^*$ 下，这就是 Bradley-Terry 模型中偏好数据的似然。回忆第 5 章「奖励建模」中我们推导过：Bradley-Terry 目标就是最大化似然，等价于最小化负对数似然，于是得到损失：
$$
\begin{aligned}
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) &= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log p(y_c \succ y_r \mid x)  \right] \\
&= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right]
\end{aligned}
$${#eq:dpo_loss_deriv4}

这就是 DPO 的损失函数，形式与 @eq:dpo_core 所示一致。
DPO 论文还给出了 Plackett-Luce 模型下目标的额外推导，但实践中用得少得多 [@rafailov2024direct]。

#### 推导 BT-DPO 的梯度

我们用 @eq:dpo_gradient 中给出的 DPO 梯度解释了模型如何学习的直觉。
要推导它，需要对 @eq:dpo_loss_deriv4 关于模型参数求梯度。

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\nabla_{\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right] $$ {#eq:dpo_grad_0}

首先，它可以重写。
我们知道 sigmoid 的导数 $\frac{d}{dx} \sigma(x) = \sigma(x)(1-\sigma(x))$、对数的导数 $\frac{d}{dx} \log x = \frac{1}{x}$，以及 sigmoid 的性质 $\sigma(-x)=1-\sigma(x)$，于是可以改写上式。

先令 $u=\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$（也就是 sigmoid 内部的表达式）。
于是有

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[\frac{\sigma'(u)}{\sigma(u)}\nabla_{\theta}u\right] $$ {#eq:dpo_grad_2}

展开它，并利用上面 sigmoid 与对数的表达式，就得到前面引入的那个梯度：

$$ -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)} - \beta\log\frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right)\left[\nabla_{\theta}\log\pi_{\theta}(y_c|x)-\nabla_{\theta}\log\pi_{\theta}(y_r|x)\right]\right] $$ {#eq:dpo_grad_3}

## 数值问题、局限与替代方案

针对 DPO 的弱点，人们提出了许多 DPO 变体。
例如，DPO 没有 rollout、也没有能对生成打分的奖励模型，因此它把每一对偏好数据同等看待。
而现实中，如第 11 章「偏好数据」所述，捕捉偏好数据的方式有很多，标签可以比二元更丰富。
已有多个算法被提出来重新平衡优化，使其不再对每一对一视同仁。

- **基于相对奖励回归的 RL（REBEL）** 引入了来自奖励模型的信号——作为被选中与被拒绝回答之间的间隔——而不只是成对偏好数据，从而更准确地求解 RLHF 问题 [@gao2024rebel]。
- **保守 DPO（cDPO）与恒等偏好优化（IPO）** 通过假设偏好数据中存在噪声来应对过拟合。cDPO 假设有 N% 的数据标注错误 [@rafailov2024direct]；IPO 则改变优化目标，柔化偏好概率，而不是直接依据标签优化 [@azar2024general]。实践上，IPO 把偏好概率换成一个非线性函数 $\Psi(q) = \log\left(\frac{q}{1-q}\right)$，从而脱离 Bradley-Terry 假设。
- **带偏移的 DPO（ODPO）** “要求偏好回答与非偏好回答的似然差大于某个偏移值” [@amini2024direct]——不再把每对数据同等看待，但代价可能是更困难的标注环境。

另一些 DPO 变体则试图通过小幅改动损失来改进学习信号，或通过降低显存占用提升效率。

- **赔率比策略优化（ORPO）** 用类似指令微调损失的方式直接更新策略模型、把模型拉向被选中回答，同时对被选中回答施加一个小的惩罚 [@hong2024reference]。这种损失函数的改动去掉了对参考模型的需求，简化了设置。理解 ORPO 最好的方式是把它看作“受 DPO 启发”，而非 DPO 的衍生。
- **简单偏好优化（SimPO）** 对 DPO 优化做了一处小改动：把对数概率取平均而非求和，或加入长度归一化，以提升性能 [@meng2025simpo]。

![DPO 中偏好位移的示意。](images/dpo_displacement.png){#fig:dpo_issue .center}

DPO 中*显而易见*的核心问题之一是：这个优化只致力于拉大被选中与被拒绝回答概率之间的间隔。
在数值上，模型会同时降低被选中和被拒绝回答的概率，但**被拒绝回答降低的幅度更大**，如 @fig:dpo_issue 所示。
直观上，这如何泛化并不清楚；但有工作提出：它会提高那些**未被触及**行为的概率——也就是语言模型能够生成、但不在后训练数据集分布内的 token [@razin2024unintentional] [@ren2024learning]。
一些简单方法可以缓解这种**偏好位移**，例如调整优化过程的 Cal-DPO [@xiao2024cal]，以及改变奖励形状的 AlphaPO [@gupta2025alphapo]。
实践中，其确切影响尚不清楚，但它指出了在线方法为何可能优于原始 DPO 的一个潜在原因。

关于 DPO 类方法性能上限低于在线（基于 RL 的）RLHF 方法，另一个被提出的主要原因是：**训练信号来自先前或其他模型生成的补全**。
DPO 的在线变体通过在训练时生成新补全并纳入偏好信号来缓解这些限制。**在线 DPO** [@guo2024direct] 从当前模型采样生成；**判别器引导的 DPO**（D2PO）[@singhal2024d2po] 用奖励模型重新打标，即时生成新的偏好数据；此外还有许多变体。

还有其他一长串 DAA 变体，例如直接纳什优化（DNO）[@rosset2024direct] 或二元分类器优化（BCO）[@jung2024binary]；但**算法选择远不如初始模型和所用数据重要** [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences]。

## 实现细节

DPO 这类 DAA 的实现方式与策略梯度优化器差别很大。
DPO 的损失取自原始实现，大致可以概括如下 [@rafailov2024direct]：

```python
# Log-probability gaps for the policy and the frozen reference model
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# Difference of log-ratios: positive when the policy
# shifts probability toward the chosen completion
logits = pi_logratios - ref_logratios

# DPO loss: negative log-sigmoid drives the policy to
# widen the gap between chosen and rejected
losses = -F.logsigmoid(beta * logits)

# Implicit rewards (detached -- used for logging only)
chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

它可以直接用于标准的语言模型训练栈，因为这些信息在模型前向传播时就已被收集（只需额外加一个参考模型）。

在大多数方面，DAA 都更简单、是体验上的改善，但它们也带来一组不同的考量。

1. **KL 散度是静态的**：在 DPO 及其他算法中，KL 散度由平衡距离惩罚与优化的 $\beta$ 参数显式设定。这是因为 DPO 朝 RLHF 目标在给定数据下的*最优*解做梯度步——它精确地走到 $\beta$ 设定的那个解。而基于 RL 的优化器则依据批次和近期数据迈步。
2. **缓存对数概率**：DPO 的简单实现在做损失计算时，会同时为策略模型和参考模型做前向传播，图方便。但这会让显存翻倍、GPU 占用上升。为了规避，可以先在整个训练数据集上算好参考模型的对数概率，之后每个批次复用这些缓存值，从而把峰值显存降低 50%。

## 使用合成偏好数据的 DAA

如今，用 DAA 做偏好微调的流行数据集大多是**合成偏好数据**：由某个前沿模型把其他模型的输出评为胜者或败者。
突出的例子包括 UltraFeedback（这一类中的第一个）[@cui2023ultrafeedback]、Tülu 3（用扩展后的 UltraFeedback 方法构建）[@lambert2024t]、SmolLM 3 的数据 [@bakouch2025smollm3]，以及随 Olmo 3 发布的 Dolci Pref 数据集 [@teamolmo2025olmo3]。

构建这些数据集的最佳实践仍在演化。
Tülu 3 及其 2024 年 11 月发布前后的数据集表明：合成成对偏好数据需要在某种意义上是**同策略**的——即部分补全应当由你正在微调的那个模型生成（同时混入更大的模型池中）。
数据的这种同策略特性，确保 DAA 会在模型实际生成所在的 token 空间中做优化——因为它的损失函数是**对比式**的，比指令微调更间接。
后来，随着 2025 年 Olmo 3 与 SmolLM 3 的发布，其他工作支持了一种不同的理论，叫 **Delta Learning**：它主张**被选中与被拒绝补全之间的差异**，比“补全究竟由哪些模型产生”对学习更重要 [@geng2025the]。
例如，在这两个被引用的模型里，被选中回答都来自 Qwen 3 32B、被拒绝回答都来自 Qwen 3 0.6B——而两位作者是各自独立、同时发展出这组配对的。

总的来说，**用 DAA 在合成偏好数据上训练，是大多数从业者应当起步的地方**——因为实现简单，而且相对于基于强化学习的偏好微调，性能表现强劲。
使用大量合成偏好数据还有其他一些小问题，比如评判模型在补全之间做比较时的偏置。
鉴于 GPT-4 这类前沿模型已知存在长度偏置 [@dubois2024length] 与偏好“与自己风格相近的输出” [@panickssery2024llm]（更多信息见第 12 章），数据集里“被选中”一侧的文本，稍微更可能来自 OpenAI 模型或其他风格相近的强模型。

本节最后，我们给出一个直觉，说明这些方法如何改变被训练模型的生成。
高层面看，多数 DAA 的优化目标都是拉大“被选中”与“被拒绝”补全概率之间的间隔（一些不太流行的算法被设计来稍微改变这些动态，但核心不变）。
如本章前面所讨论（见 @fig:dpo_issue），这通常意味着两个概率都下降，但被拒绝回答下降得更多。
序列中的每个 token 会依其“对总体偏好间隔贡献了多少”而收到不同的梯度（幅度与方向），这让优化器能够识别哪些 token 对结果最重要。

## DAA 与 RL：在线数据 vs 离线数据

粗略地说，这场争论归结为一个问题：**要把语言模型与 RLHF 对齐，我们是否需要强化学习的那套内部机制——价值函数、策略梯度等等？**
和大多数这样表述的问题一样，它过于简单。
当然，两种方法都已确立；但有必要说明根本差异与性能分布落在哪里。

多份报告得出结论：基于策略梯度与 RL 的方法优于 DPO 及其变体。
论证形式各异——有在受控数据下用不同算法训练模型的 [@ivison2024unpacking] [@xu2024dpo]，也有研究同策略数据在 RL 优化循环中作用的 [@tajwar2024preference]。
在所有这些情况下，DPO 算法都差了那么一点点。

即便存在这一性能差距，由于简单，DAA 仍然被大量用于领先模型。
DAA 提供了一个可控环境，让训练数据与其他配置的迭代可以快速进行；而鉴于数据往往比算法重要得多，用 DPO 是完全可以的。

随着主要用 RL 训练的推理模型出现，更多的投入会回到“用 RL 做偏好微调”上；长期看，这将提升 RL 基础设施的鲁棒性，并巩固 DAA 与 RL 在“从人类反馈优化”上的这一差距。

## 建议的实验

`code/direct_alignment/` 中的配套代码会在偏好数据上训练 DPO 及若干相关损失。
由于是离线的（不需要奖励模型服务，也不需要 rollout 循环），这是开始尝试偏好微调最容易上手的地方。

1. **在 UltraFeedback 上跑一个小规模 DPO。**

   ```bash
   cd code/
   uv run python -m direct_alignment.train --loss dpo --max_samples 1000
   ```

   观察 `loss`、`accuracy`、`margins`、`chosen_rewards` 和 `rejected_rewards`。
   主要的自检点是：隐式奖励间隔应当朝期望方向移动，同时模型的样本生成不应坍塌。

2. **对比 DPO、IPO 与长度归一化的 DPO。**

   ```bash
   cd code/
   uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/ipo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/dpo_norm.yaml
   ```

   对比间隔的量级与对学习率的敏感性。
   IPO 的损失不在与 DPO 相同的数值尺度上，所以应当通过 `accuracy` 与间隔行为来读它，而不是只看原始损失。

3. **谨慎尝试无参考模型的变体。**
   用各自的配置跑 SimPO 或 ORPO，然后检查训练中记录的生成样本。
   这些损失对对数概率的尺度与学习率更敏感，因此是很有价值的调试题材。

   ```bash
   cd code/
   uv run python -m direct_alignment.train --config direct_alignment/configs/simpo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/orpo.yaml
   ```

4. **先改数据，再改损失。**
   固定损失，改变 `--max_samples`、`--max_length` 或偏好数据集。
   如果结果的变化比“在 DPO 类目标之间切换”更大，那就是对偏好微调一个核心主题的经验提醒：**数据通常压过算法上的细微差异。**
