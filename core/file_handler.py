# core/file_handler.py
import os
import pypdf
import shutil
from .config import PAPERS_ROOT, IMAGES_ROOT

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

def move_file_to_category(file_path, category, file_type="paper"):
    """
    将文件移动到对应的统一资料库中。
    :param file_type: 'paper' 或 'image'，决定了文件去 PAPERS_ROOT 还是 IMAGES_ROOT
    """
    try:
        # 1. 决定去哪里 (Papers 库还是 Images 库)
        if file_type == "paper":
            base_root = PAPERS_ROOT
        elif file_type == "image":
            base_root = IMAGES_ROOT
        else:
            # 如果未知类型，默认留在原地整理
            base_root = os.path.dirname(file_path)

        # 2. 拼接目标路径
        target_dir = os.path.join(base_root, category)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        file_name = os.path.basename(file_path)
        new_path = os.path.join(target_dir, file_name)
        
        # 3. 如果源路径和目标路径完全一样，说明文件已经在对的地方了，不用动
        if os.path.abspath(file_path) == os.path.abspath(new_path):
            return new_path

        # 4. 移动文件
        shutil.move(file_path, new_path)
        
        rel_path = os.path.relpath(new_path, start=os.path.dirname(base_root))
        print(f"   [归档完成] -> {rel_path}")
        
        return new_path
    
    except Exception as e:
        print(f"   [文件移动错误] {e}")
        return file_path