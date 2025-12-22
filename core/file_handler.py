# core/file_handler.py
import os
import pypdf
import shutil

def read_pdf_chunks(pdf_path):
    chunks = []
    file_name = os.path.basename(pdf_path)
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text) > 50: # 忽略内容太少的页面
                chunks.append({
                    "text": text, "page": i + 1, "source": file_name, "path": pdf_path
                })
    except Exception as e:
        print(f"读取 PDF 失败: {e}")
    return chunks

def move_file_to_category(file_path, category, base_dir="papers"):
    """把文件移动到分类文件夹"""
    target_dir = os.path.join(base_dir, category)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_name = os.path.basename(file_path)
    new_path = os.path.join(target_dir, file_name)
    shutil.move(file_path, new_path)
    print(f"文件已移动到: {new_path}")
    return new_path