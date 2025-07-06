import pdfplumber

pdf_path = r""
save_path = r""

with pdfplumber.open(pdf_path) as pdf, open(save_path, "w", encoding="utf-8") as f:
    for i, page in enumerate(pdf.pages):
        # 일반 텍스트 추출
        f.write(f"[Page {i+1} - Text]\n")
        f.write(page.extract_text() or "")
        f.write("\n\n")

        # 표 추출
        tables = page.extract_tables()
        for j, table in enumerate(tables):
            f.write(f"[Page {i+1} - Table {j+1}]\n")
            for row in table:
                row_str = "\t".join(cell or "" for cell in row)
                f.write(row_str + "\n")
            f.write("\n")

print(f"[✅ 완료] 텍스트 + 표 추출 저장됨 → {save_path}")
