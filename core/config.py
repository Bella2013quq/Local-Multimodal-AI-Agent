# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# api
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# 数据库和模型路径配置
DB_PATH = "./my_knowledge_base"
MODEL_PATH_CLIP = "openai/clip-vit-base-patch32"

# 资料库路径配置 
LIBRARY_ROOT = "./"  

# 定义论文和图片资料库
PAPERS_ROOT = os.path.join(LIBRARY_ROOT, "papers")
IMAGES_ROOT = os.path.join(LIBRARY_ROOT, "images")

# 自动创建根目录
if not os.path.exists(PAPERS_ROOT): os.makedirs(PAPERS_ROOT)
if not os.path.exists(IMAGES_ROOT): os.makedirs(IMAGES_ROOT)