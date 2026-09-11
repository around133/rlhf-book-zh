#!/usr/bin/env python3
"""给 pandoc 产出的 DOCX 补上页脚。

pandoc 的 docx 写出器不会保留参考文档里的 sectPr，因此参考文档中定义的页脚
虽然在包内（word/footer1.xml 会被一并复制），却没有任何节引用它。

本脚本对构建产物做三件事：
  1. 在 <w:body> 末尾插入带 footerReference 的 sectPr（同时设定页边距）
  2. 在 word/_rels/document.xml.rels 中加入页脚关系
  3. 在 [Content_Types].xml 中登记 footer1.xml

用法：
    python zh/tools/add-docx-footer.py zh/build/rlhf-book-zh.docx
"""

import os
import re
import shutil
import sys
import tempfile
import zipfile

R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
RID = "rIdFooterZh"

SECTPR = (
    "<w:sectPr>"
    '<w:footerReference w:type="default" r:id="%s"/>'
    '<w:pgSz w:w="12240" w:h="15840"/>'
    '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
    'w:header="720" w:footer="720" w:gutter="0"/>'
    "</w:sectPr>" % RID
)

FOOTER_REL = ('<Relationship Id="%s" Type="%s/footer" Target="footer1.xml"/>'
              % (RID, R_NS))

FOOTER_CT = ('<Override PartName="/word/footer1.xml" ContentType='
             '"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')


def patch(path):
    zin = zipfile.ZipFile(path, "r")
    items = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    doc = items["word/document.xml"].decode("utf-8")
    if "footerReference" in doc:
        print("已存在页脚引用，跳过")
        return
    if re.search(r"<w:sectPr[ >].*?</w:sectPr>", doc, re.S):
        doc = re.sub(r"<w:sectPr[ >].*?</w:sectPr>", SECTPR, doc, flags=re.S)
    else:
        doc = doc.replace("</w:body>", SECTPR + "</w:body>")
    items["word/document.xml"] = doc.encode("utf-8")

    rels = items["word/_rels/document.xml.rels"].decode("utf-8")
    if RID not in rels:
        rels = rels.replace("</Relationships>", FOOTER_REL + "</Relationships>")
    items["word/_rels/document.xml.rels"] = rels.encode("utf-8")

    ct = items["[Content_Types].xml"].decode("utf-8")
    if "footer1.xml" not in ct:
        ct = ct.replace("</Types>", FOOTER_CT + "</Types>")
    items["[Content_Types].xml"] = ct.encode("utf-8")

    if "word/footer1.xml" not in items:
        print("警告：包内没有 word/footer1.xml，请先运行 make-docx-reference.py")

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, data in items.items():
            zout.writestr(n, data)
    shutil.move(tmp, path)
    print("已为 %s 补上页脚" % os.path.basename(path))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        patch(p)
