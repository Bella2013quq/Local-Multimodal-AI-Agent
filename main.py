import argparse
import os
import sys

# 引入核心模块
from core.ai_handler import AIHandler
from core.db_handler import DatabaseHandler
from core.file_handler import read_pdf_chunks, move_file_to_category

from dotenv import load_dotenv
load_dotenv()  

from core.config import GEMINI_API_KEY

def process_paper(ai, db, file_path, topics):
    """处理单篇 PDF 论文的逻辑"""
    filename = os.path.basename(file_path)
    print(f"\n[PDF] 正在处理: {filename}")
    
    # 1. 去重检查
    if db.check_paper_exists(filename):
        print(f"   [跳过]: 数据库中已存在该论文")
        return

    # 2. 读取并分块
    chunks = read_pdf_chunks(file_path)
    if not chunks: 
        print("   [跳过]: PDF 读取为空或失败")
        return

    # 3. 生成向量
    print("   正在生成文本向量...")
    try:
        embeddings = [ai.get_gemini_embedding(c['text']) for c in chunks]
    except Exception as e:
        print(f"   [错误] 向量生成失败: {e}")
        return
    
    # 4. 智能分类 & 移动文件
    print("   Gemini 正在阅读摘要并分类...")
    first_page_text = chunks[0]['text'][:1000]
    prompt = f"请阅读以下论文摘要，并从这些类别中选择最合适的一个：[{topics}]。只返回类别名称，不要标点符号。\n\n摘要：{first_page_text}"
    
    category = "Uncategorized"
    try:
        category = ai.chat_with_gemini(prompt).strip()
        category = category.replace("'", "").replace('"', "").replace(".", "")
        print(f"   分类结果: {category}")
        
        new_path = move_file_to_category(file_path, category, file_type="paper")
        if new_path:
            file_path = new_path 
            
    except Exception as e:
        print(f"   [警告] 分类或移动失败: {e}")

    # 5. 入库 必须在移动文件之后，确保路径是最新的
    db.add_paper_chunks(chunks, embeddings, moved_path=file_path, category=category)
    print("   论文处理完成。")


def process_image(ai, db, file_path, topics="Screenshot,Diagram,Photo,Art,Infographic,Other"):
    """处理单张图片的逻辑"""
    filename = os.path.basename(file_path)
    print(f"\n[IMG] 正在处理: {filename}")
    
    # 1. 去重检查
    if db.check_image_exists(filename):
        print(f"   [跳过]: 数据库中已存在该图片")
        return

    # 2. CLIP 向量
    try:
        clip_vec = ai.get_clip_embedding(file_path)
    except Exception as e:
        print(f"   [跳过]: CLIP处理失败 {e}")
        return

    # 3. Gemini 描述
    print("   Gemini 正在观察图片...")
    try:
        desc = ai.get_image_description(file_path)
        print(f"   描述: {desc[:30]}...")
    except Exception as e:
        print(f"   [跳过]: Gemini描述生成失败 {e}")
        return

    # 4. 智能分类 & 移动文件
    print(f"   正在智能分类 (选项: {topics})...")
    category = "Uncategorized"
    try:
        classify_prompt = (
            f"基于以下图片描述，将图片归类为[{topics}]中的一项。\n"
            f"只返回类别名称，不要标点符号。\n\n"
            f"图片描述：{desc}"
        )
        category = ai.chat_with_gemini(classify_prompt).strip()
        category = category.replace("'", "").replace('"', "").replace(".", "")
        print(f"   分类结果: {category}")
        
        new_path = move_file_to_category(file_path, category, file_type="image")
        if new_path:
            file_path = new_path
            
    except Exception as e:
        print(f"   [警告] 分类或移动失败: {e}")
    
    # 5. 描述向量化
    gemini_vec = ai.get_gemini_embedding(desc)

    # 6. 入库
    db.add_image(file_path, clip_vec, desc, gemini_vec, category=category)
    print("   图片处理完成。")


# 主程序

def main():
    parser = argparse.ArgumentParser(description="Gemini 本地多模态助手")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 1. 添加论文
    add_p = subparsers.add_parser("add_paper", help="添加单篇论文")
    add_p.add_argument("path", help="PDF文件路径")
    add_p.add_argument("--topics", default="Reinforcement_Learning,Spatio-Temporal_Mining,Multimodal_Learning", help="分类选项")

    # 2. 搜论文 (QA模式)
    search_p = subparsers.add_parser("search_paper", help="搜论文 (问答模式)")
    search_p.add_argument("query", help="问题")

    # 3. 文件索引 (列表模式)
    list_p = subparsers.add_parser("list_papers", help="根据主题列出相关论文文件")
    list_p.add_argument("topic", help="主题或关键词")

    # 4. 添加图片
    add_i = subparsers.add_parser("add_image", help="添加单张图片")
    add_i.add_argument("path", help="图片路径")
    add_i.add_argument("--topics", default="Model_Architecture,Performance_Plot,Table,Qualitative_Visualization,Algorithm_Math", help="分类选项")

    # 5. 搜图片
    search_i = subparsers.add_parser("search_image", help="搜图片")
    search_i.add_argument("query", help="描述")

    # 6. 搜图并提问
    ask_i = subparsers.add_parser("ask_image", help="搜图并提问")
    ask_i.add_argument("desc", help="用于定位图片的描述")
    ask_i.add_argument("question", help="基于图片想问的具体问题")

    # 7. 批量整理
    batch_p = subparsers.add_parser("batch_ingest", help="批量扫描文件夹处理所有文件")
    batch_p.add_argument("folder", help="文件夹路径")
    batch_p.add_argument("--topics", default="Reinforcement_Learning,Spatio-Temporal_Mining,Multimodal_Learning", help="论文分类选项")
    batch_p.add_argument("--img_topics", default="Model_Architecture,Performance_Plot,Table,Qualitative_Visualization,Algorithm_Math", help="图片分类选项")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化
    try:
        ai = AIHandler()
        db = DatabaseHandler()
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        return

    # 逻辑分支

    # 1. 单个处理论文
    if args.command == "add_paper":
        if os.path.exists(args.path):
            process_paper(ai, db, args.path, args.topics)
        else:
            print("[错误] 文件不存在")

    # 2. 单个处理图片
    elif args.command == "add_image":
        if os.path.exists(args.path):
            process_image(ai, db, args.path, topics=args.topics)
        else:
            print("[错误] 文件不存在")

    # 3. 批量处理
    elif args.command == "batch_ingest":
        folder_path = args.folder
        if not os.path.isdir(folder_path):
            print(f"[错误] '{folder_path}' 不是一个有效的文件夹")
            return
        
        print(f"开始扫描文件夹: {folder_path} ...")
        
        count_pdf = 0
        count_img = 0
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith('.'): continue

                full_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                if ext == '.pdf':
                    process_paper(ai, db, full_path, args.topics)
                    count_pdf += 1
                elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    process_image(ai, db, full_path, topics=args.img_topics)
                    count_img += 1
        
        print(f"\n批量处理完成！PDF: {count_pdf}, IMG: {count_img}")

    # 4. 搜论文 (QA)
    elif args.command == "search_paper":
        print(f"正在检索: {args.query}")
        query_vec = ai.get_gemini_embedding(args.query)
        results = db.search_paper(query_vec, n_results=3)
        
        docs = results['documents'][0]
        metas = results['metadatas'][0]

        if not docs:
            print("[提示] 无相关信息。")
            return

        print("\n--- 参考片段")
        for i, meta in enumerate(metas):
            # 防止旧数据报错
            src = meta.get('source', '未知')
            page = meta.get('page', '?')
            print(f"[{i+1}] {src} (P.{page})")

        print("\nGemini 回答:")
        context = "\n\n".join(docs)
        prompt = f"你是一个学术助手。基于以下参考资料回答用户问题：{args.query}\n\n参考资料：\n{context}"
        answer = ai.chat_with_gemini(prompt)
        print(answer)

    elif args.command == "list_papers":
        print(f"正在索引主题: '{args.topic}' ...")
        
        # 1. 向量搜索 
        query_vec = ai.get_gemini_embedding(args.topic)
        results = db.search_paper(query_vec, n_results=15)
        
        metas = results['metadatas'][0]
        if not metas:
            print("[提示] 未找到相关论文。")
            return

        # 2. 聚合候选文件
        candidate_files = {}
        for meta in metas:
            path = meta.get('path', meta.get('source', 'Unknown'))
            filename = os.path.basename(path)
            category = meta.get('category', 'Unknown')
            
            # 记录下来，稍后发给 Gemini
            if filename not in candidate_files:
                candidate_files[filename] = {
                    "category": category,
                    "source_path": path
                }

        print(f"向量库初筛找到 {len(candidate_files)} 篇候选论文，正在进行 AI 重排序...")

        # 3. 构造 Prompt 进行重排序 
        # 将候选列表转为文本
        candidates_text = ""
        for i, (fname, info) in enumerate(candidate_files.items()):
            candidates_text += f"{i+1}. 文件名: {fname} (分类: {info['category']})\n"

        rerank_prompt = (
            f"用户正在寻找关于 '{args.topic}' 的论文。\n"
            f"向量数据库找出了以下候选文件，请你判断哪些文件真正与主题高度相关，并按相关性从高到低排序。\n"
            f"请排除掉明显不相关的文件（例如只是提到了关键词但核心主题不符的）。\n\n"
            f"候选列表：\n{candidates_text}\n\n"
            f"请输出一个简洁的列表，格式如下：\n"
            f"1. [相关度: 高/中/低] 文件名 - 一句话解释为什么相关\n"
        )

        # 4. 让 Gemini 进行最终裁决
        try:
            ranking_result = ai.chat_with_gemini(rerank_prompt)
            print("\n" + "="*30)
            print(f"Gemini 智能筛选结果 (主题: {args.topic})")
            print("="*30)
            print(ranking_result)
            print("="*30)
        except Exception as e:
            print(f"[错误] 重排序失败: {e}")
            # 如果 AI 失败，回退到简单的列表展示
            for fname in candidate_files:
                print(f"- {fname}")

    # 6. 搜图片
    elif args.command == "search_image":
        print(f"正在进行双模搜索: '{args.query}'\n")
        
        # A. Gemini
        gemini_query_vec = ai.get_gemini_embedding(args.query)
        gemini_results = db.search_image_desc(gemini_query_vec, n_results=3)
        if gemini_results['metadatas'][0]:
            print("[语义匹配]")
            for i, meta in enumerate(gemini_results['metadatas'][0]):
                desc = meta.get('desc', '')[:30].replace('\n', ' ')
                cat = meta.get('category', '')
                print(f"  {i+1}. [{cat}] {os.path.basename(meta['path'])} | {desc}...")
        
        # B. CLIP
        try:
            clip_query_vec = ai.get_clip_text_embedding(args.query)
            clip_results = db.search_image_clip(clip_query_vec, n_results=3)
            if clip_results['metadatas'][0]:
                print("\n[视觉匹配]")
                for i, meta in enumerate(clip_results['metadatas'][0]):
                    print(f"  {i+1}. {os.path.basename(meta['path'])}")
        except:
            pass

    # 7. 搜图并提问
    elif args.command == "ask_image":
        print(f"正在定位图片: '{args.desc}'...")
        best_path = None
        
        # 1. Gemini
        g_vec = ai.get_gemini_embedding(args.desc)
        res_g = db.search_image_desc(g_vec, n_results=1)
        if res_g['metadatas'][0]:
            best_path = res_g['metadatas'][0][0]['path']
            print(f"[Gemini] 锁定: {os.path.basename(best_path)}")
        
        # 2. CLIP 
        if not best_path:
            try:
                c_vec = ai.get_clip_text_embedding(args.desc)
                res_c = db.search_image_clip(c_vec, n_results=1)
                if res_c['metadatas'][0]:
                    best_path = res_c['metadatas'][0][0]['path']
                    print(f"[CLIP] 锁定: {os.path.basename(best_path)}")
            except: pass
            
        if best_path and os.path.exists(best_path):
            print(f"Gemini 正在回答: {args.question}")
            print(ai.chat_with_image(best_path, args.question))
        else:
            print("[错误] 未找到图片。")

if __name__ == "__main__":
    main()