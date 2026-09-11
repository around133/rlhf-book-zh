#!/usr/bin/env python3
"""从官方 DOCX 参考模板生成中文版参考模板。

官方模板 book/templates/docx.docx 用的是主题字体（minorHAnsi 等），没有显式指定
中文字体，也没有页眉页脚。本脚本产出一个 zh/templates/docx-zh.docx：

  1. 正文西文用 Times New Roman、中文用宋体（SimSun）
  2. 各级标题中文用黑体（SimHei）
  3. 页脚居中写「翻译人：晏一夫」

用法：
    python zh/tools/make-docx-reference.py
"""

import io
import os
import re
import shutil
import zipfile

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(BASE, "book", "templates", "docx.docx")
DST_DIR = os.path.join(BASE, "zh", "templates")
DST = os.path.join(DST_DIR, "docx-zh.docx")

FOOTER_TEXT = "翻译人：北航 控制科学与工程博士 晏一夫　Yanyifu633@buaa.edu.cn"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

FOOTER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:ftr xmlns:w="%s">'
    "<w:p>"
    '<w:pPr><w:jc w:val="center"/><w:rPr><w:sz w:val="18"/><w:color w:val="808080"/></w:rPr></w:pPr>'
    "<w:r><w:rPr><w:sz w:val=\"18\"/><w:color w:val=\"808080\"/></w:rPr>"
    "<w:t>%s</w:t></w:r>"
    "</w:p>"
    "</w:ftr>" % (W, FOOTER_TEXT)
)


def latin_fonts(ascii_font, east_asia):
    """构造一个 rFonts 元素。"""
    return ('<w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
            % (ascii_font, ascii_font, east_asia, ascii_font))


def patch_styles(xml):
    # 1) 文档默认字体：西文 Times New Roman，中文宋体
    xml = re.sub(
        r"<w:rFonts[^/]*/>",
        lambda m: latin_fonts("Times New Roman", "SimSun"),
        xml,
        count=1,  # 只改 docDefaults 里的那个
    )

    # 2) 标题类样式的中文用黑体
    def heading_font(m):
        return latin_fonts("Times New Roman", "SimHei")

    def patch_style(body):
        return re.sub(r"<w:rFonts[^/]*/>", heading_font, body, count=1)

    for sid in ("Title", "Subtitle", "Heading1", "Heading2", "Heading3",
                "Heading4", "Heading5", "Heading6", "Heading7", "Heading8",
                "Heading9", "TOCHeading"):
        pat = re.compile(
            r'(<w:style [^>]*w:styleId="%s"[^>]*>)(.*?)(</w:style>)' % sid, re.S)
        xml = pat.sub(lambda m: m.group(1) + patch_style(m.group(2)) + m.group(3), xml)

    # 3) 页面语言改为中文，避免拼写检查与断行按英文处理
    xml = xml.replace(
        '<w:lang w:val="en-US" w:eastAsia="en-US" w:bidi="ar-SA" />',
        '<w:lang w:val="en-US" w:eastAsia="zh-CN" w:bidi="ar-SA" />')
    return xml


def patch_document(xml):
    """确保文档级 sectPr 引用页脚。"""
    sect = (
        "<w:sectPr>"
        '<w:footerReference w:type="default" r:id="rIdFooterZh"/>'
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
    )
    if re.search(r"<w:sectPr[ >].*?</w:sectPr>", xml, re.S):
        xml = re.sub(r"<w:sectPr[ >].*?</w:sectPr>",
                     sect.replace("<w:sectPr>", '<w:sectPr>', 1), xml, flags=re.S)
    else:
        # 在 </w:body> 前插入（sectPr 必须是 body 的最后一个子元素）
        xml = xml.replace("</w:body>", sect + "</w:body>")
    return xml


def patch_rels(xml):
    add = ('<Relationship Id="rIdFooterZh" Type="%s/footer" '
           'Target="footer1.xml"/>' % R)
    if "rIdFooterZh" in xml:
        return xml
    return xml.replace("</Relationships>", add + "</Relationships>")


def patch_content_types(xml):
    add = ('<Override PartName="/word/footer1.xml" ContentType='
           '"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')
    if "footer1.xml" in xml:
        return xml
    return xml.replace("</Types>", add + "</Types>")


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    zin = zipfile.ZipFile(SRC, "r")
    items = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    items["word/styles.xml"] = patch_styles(
        items["word/styles.xml"].decode("utf-8")).encode("utf-8")
    items["word/document.xml"] = patch_document(
        items["word/document.xml"].decode("utf-8")).encode("utf-8")
    items["word/_rels/document.xml.rels"] = patch_rels(
        items["word/_rels/document.xml.rels"].decode("utf-8")).encode("utf-8")
    items["[Content_Types].xml"] = patch_content_types(
        items["[Content_Types].xml"].decode("utf-8")).encode("utf-8")
    items["word/footer1.xml"] = FOOTER_XML.encode("utf-8")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, data in items.items():
            zout.writestr(n, data)

    print("已生成: %s (%.0f KB)" % (DST, os.path.getsize(DST) / 1024))
    print("  正文中文: 宋体 / 标题中文: 黑体 / 页脚: %s" % FOOTER_TEXT)


if __name__ == "__main__":
    main()
