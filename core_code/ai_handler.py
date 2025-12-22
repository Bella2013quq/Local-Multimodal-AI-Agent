# core/ai_handler.py
import os
# 如果你需要代理，保留这些；不需要则注释掉
os.environ['HTTP_PROXY']  = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HF_HOME'] = r"./model"

import google.generativeai as genai
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import PIL.Image
from .config import GEMINI_API_KEY, MODEL_PATH_CLIP

class AIHandler:
    def __init__(self):
        print("正在初始化 AI 模型...")
        
        # 1. 配置 Gemini
        print("正在加载 Gemini 模型...")
        if not GEMINI_API_KEY or "AIza" not in GEMINI_API_KEY:
            raise ValueError("请先在 core/config.py 中填入正确的 GEMINI_API_KEY")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_flash = genai.GenerativeModel('gemini-2.5-flash') 
        print("Gemini 模型就绪")
        
        # 2. 配置本地 CLIP
        print("正在加载 CLIP 模型...")
        try:
            self.clip_model = CLIPModel.from_pretrained(MODEL_PATH_CLIP)
            self.clip_processor = CLIPProcessor.from_pretrained(MODEL_PATH_CLIP)
            print("CLIP 模型就绪")
        except Exception as e:
            print(f"CLIP 模型加载失败: {e}")
            print("请检查 MODEL_PATH_CLIP 路径是否正确，或者网络是否能连接 HuggingFace")

    def get_gemini_embedding(self, text):
        """Gemini: 把文字变成 768维 向量"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="semantic_similarity"
            )
            return result['embedding']
        except Exception as e:
            print(f"Gemini Embedding 失败: {e}")
            return []

    def get_clip_embedding(self, image_path):
        """CLIP: 把图片变成 512维 向量"""
        try:
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            # 归一化并转列表
            return image_features.detach().numpy().flatten().tolist()
        except Exception as e:
            print(f"CLIP 图片向量化失败: {e}")
            return []

    def get_clip_text_embedding(self, text):
        """CLIP: 把文本变成 512维 向量 (用于视觉搜索)"""
        try:
            # 截断过长的文本，因为 CLIP 对长度敏感
            inputs = self.clip_processor(text=[text[:77]], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            return text_features.detach().numpy().flatten().tolist()
        except Exception as e:
            print(f"CLIP 文本向量化失败: {e}")
            return []

    def get_image_description(self, image_path):
        """让 Gemini 看图说话"""
        image = PIL.Image.open(image_path)
        prompt = "请详细描述这张图片的内容，包括主体、颜色、动作、文字信息(OCR)及整体氛围。不要分段，直接输出一段中文描述。"
        response = self.gemini_flash.generate_content([prompt, image])
        return response.text

    def chat_with_gemini(self, prompt):
        """普通对话"""
        response = self.gemini_flash.generate_content(prompt)
        return response.text
    
    def chat_with_image(self, image_path, user_question):
        """图片问答"""
        try:
            img = PIL.Image.open(image_path)
            # [修正] 这里原来写成了 self.model，应改为 self.gemini_flash
            response = self.gemini_flash.generate_content([user_question, img])
            return response.text
        except Exception as e:
            return f"图片问答出错: {str(e)}"