#!/usr/bin/env python3
"""把中文译文里的直引号改成中文弯引号，并把 --> 换成 →。

为什么需要：中文排版必须使用 “ ” 而不是 "。直引号在 xeCJK 环境下
渲染不稳定，会出现成对的右引号 ”...”。

会跳过：
  - YAML front matter（否则会破坏 metadata 语法和 URL）
  - 围栏代码块 ``` ... ```
  - 行内代码 `...`

用法：
    python zh/tools/fix-quotes.py zh/chapters/03-training-overview.md
"""

import io
import re
import sys


def split_front_matter(lines):
    """返回 (front_matter_end_index, body_start_index)。没有 front matter 时返回 (-1, 0)。"""
    start = None
    for i, l in enumerate(lines):
        if l.strip() == "---":
            start = i
            break
    if start is None:
        return -1, 0
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "---":
            return j, j + 1
    return -1, 0


def process(path):
    raw = io.open(path, encoding="utf-8").read()
    lines = raw.split("\n")
    fm_end, body_start = split_front_matter(lines)

    out = []
    in_code = False
    n_quotes = 0
    n_arrows = 0
    # 引号的开关状态跨行保持，否则跨行的引文（例如一首诗被拆成多行引用块）
    # 会在每一行重新从“左引号”开始，产生两个左引号而无右引号。
    open_q = True

    for i, line in enumerate(lines):
        if i < body_start:
            out.append(line)
            continue
        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue

        parts = re.split(r"(`[^`]*`)", line)
        for k, seg in enumerate(parts):
            if seg.startswith("`") and seg.endswith("`") and len(seg) > 1:
                continue
            n_quotes += seg.count('"')
            buf = []
            for ch in seg:
                if ch == '"':
                    buf.append("\u201c" if open_q else "\u201d")
                    open_q = not open_q
                else:
                    buf.append(ch)
            parts[k] = "".join(buf)
        line2 = "".join(parts)
        # 只在“不是 HTML 注释”的行上把 --> 换成 →。
        # 否则 <!-- ... --> 的结束符会被破坏，导致注释不闭合、
        # pandoc 把后续大段正文当作注释吞掉。
        if "<!--" not in line2 and line2.strip() != "-->":
            n_arrows += line2.count("-->")
            line2 = line2.replace("-->", "\u2192")
        out.append(line2)

    io.open(path, "w", encoding="utf-8").write("\n".join(out))
    text = io.open(path, encoding="utf-8").read()
    return n_quotes, n_arrows, text.count('"'), text.count("\u201c"), text.count("\u201d")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        q, a, remain, lq, rq = process(p)
        print("%s: 引号 %d 处, 箭头 %d 处, 剩余直引号 %d, 弯引号 %d/%d" % (p, q, a, remain, lq, rq))
