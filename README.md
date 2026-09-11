# RLHF Book 中文版

Nathan Lambert《Reinforcement Learning from Human Feedback》的中文译本。

> **非官方翻译。** 本译本由译者独立完成，与原作者及出版方无隶属关系。
> 原作者 Nathan Lambert 已知悉本项目，并表示欢迎分享（参见下方「致谢」）。

## 下载

| 文件 | 说明 |
|---|---|
| **[rlhf-book-zh.pdf](rlhf-book-zh.pdf)** | 全书 PDF，204 页，12 MB |
| **[rlhf-book-zh.docx](rlhf-book-zh.docx)** | Word 版，12 MB |

## 原文

| | |
|---|---|
| 书名 | *Reinforcement Learning from Human Feedback* |
| 作者 | Nathan Lambert |
| 在线版 | <https://rlhfbook.com> |
| 配套课程 | <https://rlhfbook.com/course> |
| 源代码仓库 | <https://github.com/natolambert/rlhf-book> |

这本书系统讲解了语言模型后训练的完整链路：指令微调、奖励建模、强化学习
（PPO / GRPO / GSPO / CISPO）、可验证奖励的强化学习（RLVR）、直接对齐算法
（DPO 及其变体）、拒绝采样、合成数据与蒸馏，以及过度优化、正则化、评估等实践议题。

## 本译本的内容

覆盖英文版**第 1–17 章正文、附录 A/B/C 与参考文献**，依据 2026 年 9 月的仓库主分支翻译
（对应英文版 v0.11 及之后的增补）。

| | |
|---|---|
| 页数 | 204 页（英文原版 239 页） |
| 图片 | 53 张 |
| 表格 | 9 张 |
| 参考文献 | 463 条 |
| 编号公式 | 130 余个 |

翻译时遵循的几条原则：

- **公式、代码、图表标签、引用键一律保持原样**，因此交叉引用、图表编号与文献列表
  全部自动生成，与英文版一一对应
- **英文示例文本保留原文**（模型补全对比、聊天模板的 token 序列、JSON/XML 数据、
  提示词模板等），因为这些内容脱离英文语境就失去了示范意义
- 专业术语首次出现时给出英文原词，如「奖励模型（reward model）」
- 中文排版细节：图表编号与公式引用均为中文（「图 1」「表 1」「见式 1」），
  正文宋体、标题黑体，数学与英文沿用原文的 Latin Modern 字体

## 构建

译文的源文件是本仓库 `chapters/` 下的 Markdown，结构与英文版一一对应。

**注意：** 排版依赖英文仓库的 LaTeX 模板与参考文献库，因此完整重建需要先克隆上游仓库，
再把本仓库内容放入其 `zh/` 目录：

```bash
git clone https://github.com/natolambert/rlhf-book.git
cd rlhf-book
# 将本仓库的 chapters/ metadata*.yml build.ps1 tools/ 复制到 zh/ 对应位置
```

环境依赖：

- `pandoc` 3.8 及以上
- `TeX Live`（含 `xelatex` 与 `ctex`）
- `pandoc-crossref`（版本需与 pandoc 匹配）
- 中文字体：宋体（SimSun）、黑体（SimHei）、楷体（KaiTi）

编译：

```powershell
.\zh\build.ps1           # 编译各章为单章 PDF
.\zh\build.ps1 -Book     # 装配整书 PDF
.\zh\build.ps1 -Docx     # 装配 Word 版
```

## 致谢

感谢 Nathan Lambert 将这本书与配套课程完整开源。对中文读者而言，这是一份难得
系统、且紧跟前沿的后训练材料。

译者曾就本译本致信作者，作者回复：

> This looks great [...] Feel free to share this, and if you host it online
> somewhere I'll add a link to this ecosystem section.

## 许可

- **原文内容**：CC BY-NC-SA 4.0（署名 — 非商业性使用 — 相同方式共享），
  著作权归 Nathan Lambert 所有
  <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- **本译本**：以相同协议 CC BY-NC-SA 4.0 提供
- **构建脚本**（`build.ps1`、`tools/`）：MIT

**本译本仅供学习与研究，禁止任何商业性使用。**

## 译者

北航 控制科学与工程博士 晏一夫
`Yanyifu633@buaa.edu.cn`
