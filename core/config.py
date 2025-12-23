# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# api
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# 数据库和模型路径配置
DB_PATH = "./my_knowledge_base"
MODEL_PATH_CLIP = "openai/clip-vit-base-patch32"