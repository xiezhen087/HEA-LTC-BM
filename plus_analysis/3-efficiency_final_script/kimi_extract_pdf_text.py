# -*- coding: utf-8 -*-
"""Extract full text from the 4 literature PDFs into txt files for analysis."""
import os
import sys
from pypdf import PdfReader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "final_script", "extracted_text")
os.makedirs(OUT_DIR, exist_ok=True)

PDFS = [
    "13-JAC-多目标优化带实验-轻质高熵合金-提高硬度降低密度.pdf",
    "14-JAC-多目标优化-屈服强度+极限抗压强度+弹性模量+断裂应变pdf.pdf",
    "27-Advanced Science-1区(15.1)-机械硬度与磁性软化.pdf",
    "8-npj计算材料-微观结构稳定性+γ ' 溶剂温度+γ ' 体积分数+密度+加工窗口+冻结范围和抗氧化性.pdf",
]

for i, name in enumerate(PDFS, start=1):
    path = os.path.join(BASE, name)
    out = os.path.join(OUT_DIR, f"paper{i}.txt")
    try:
        reader = PdfReader(path)
        n_pages = len(reader.pages)
        chunks = []
        for pno, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception as e:
                txt = f"[EXTRACTION ERROR on page {pno}: {e}]"
            chunks.append(f"\n===== PAGE {pno} =====\n{txt}")
        full = "".join(chunks)
        with open(out, "w", encoding="utf-8") as f:
            f.write(full)
        print(f"paper{i}: {name} -> {n_pages} pages, {len(full)} chars -> {out}")
    except Exception as e:
        print(f"paper{i}: {name} FAILED: {e}")
        sys.exit(1)
