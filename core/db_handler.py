import chromadb
import os
from .config import DB_PATH

class DatabaseHandler:
    def __init__(self):
        print(f"正在连接数据库: {DB_PATH}")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        # 1. 论文库 (Gemini 768维)
        self.paper_collection = self.client.get_or_create_collection(
            name="paper_db", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # 2. 图片描述库 (Gemini 768维)
        self.image_desc_collection = self.client.get_or_create_collection(
            name="image_desc_db", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # 3. 视觉库 (CLIP 512维)
        self.visual_collection = self.client.get_or_create_collection(
            name="visual_db", 
            metadata={"hnsw:space": "cosine"}
        )

    def check_paper_exists(self, filename):
        """检查论文是否已存在"""
        existing = self.paper_collection.get(
            where={"source": filename}, 
            limit=1
        )
        return len(existing['ids']) > 0

    def check_image_exists(self, filename):
        """检查图片是否已存在"""
        existing = self.visual_collection.get(
            ids=[f"img_clip_{filename}"]
        )
        return len(existing['ids']) > 0

    def add_paper_chunks(self, chunks, embeddings, moved_path=None, category="Uncategorized"):
        """
        存入论文切片
        :param moved_path: 文件移动后的新路径 
        :param category: 论文分类 
        """
        ids = [f"{c['source']}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
        
        metadatas = []
        for c in chunks:
            final_path = moved_path if moved_path else c['path']
            
            metadatas.append({
                "source": c['source'], 
                "page": c['page'], 
                "path": final_path,   
                "category": category  
            })

        documents = [c['text'] for c in chunks]
        
        self.paper_collection.upsert(
            ids=ids, 
            embeddings=embeddings, 
            metadatas=metadatas, 
            documents=documents
        )
        print(f"已更新/存入 {len(chunks)} 个片段到论文库 (分类: {category})")

    def add_image(self, image_path, clip_vec, description, gemini_vec, category="Uncategorized"):
        """双路存入图片"""
        file_name = os.path.basename(image_path)
        
        # 1. 存入视觉库 (CLIP)
        self.visual_collection.upsert(
            ids=[f"img_clip_{file_name}"], 
            embeddings=[clip_vec], 
            metadatas=[{
                "path": image_path, 
                "category": category
            }]
        )
        
        # 2. 存入图片描述库 (Gemini)
        self.image_desc_collection.upsert(
            ids=[f"img_desc_{file_name}"], 
            embeddings=[gemini_vec], 
            documents=[description], 
            metadatas=[{
                "path": image_path, 
                "desc": description, 
                "category": category
            }] 
        )
        print(f"图片已双路更新/存入: {file_name} (分类: {category})")

    # --- 搜索接口保持不变 ---
    def search_paper(self, query_vec, n_results=3):
        return self.paper_collection.query(query_embeddings=[query_vec], n_results=n_results)

    def search_image_desc(self, query_vec, n_results=3):
        return self.image_desc_collection.query(query_embeddings=[query_vec], n_results=n_results)

    def search_image_clip(self, clip_vec, n_results=3):
        return self.visual_collection.query(query_embeddings=[clip_vec], n_results=n_results)