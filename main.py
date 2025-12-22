import argparse
import os
import sys

# 引入核心模块
from core.ai_handler import AIHandler
from core.db_handler import DatabaseHandler
from core.file_handler import read_pdf_chunks, move_file_to_category

# 核心处理函数

def process_paper(ai, db, file_path, topics):
    """处理单篇 PDF 论文的逻辑"""
    filename = os.path.basename(file_path)
    print(f"\n[PDF] 正在处理: {filename}")
    
    # 去重，检查数据库中是否已存在
    if db.check_paper_exists(filename):
        print(f"[跳过]: 数据库中已存在该论文")
        return

    # 1. 读取并分块
    chunks = read_pdf_chunks(file_path)
    if not chunks: 
        print("[跳过]: PDF 读取为空或失败")
        return

    # 2. 生成向量
    print("正在生成文本向量...")
    try:
        embeddings = [ai.get_gemini_embedding(c['text']) for c in chunks]
    except Exception as e:
        print(f"[错误] 向量生成失败: {e}")
        return
    
    # 3. 入库
    db.add_paper_chunks(chunks, embeddings)

    # 4. 智能分类
    print("Gemini 正在阅读摘要并分类...")
    first_page_text = chunks[0]['text'][:1000]
    prompt = f"请阅读以下论文摘要，并从这些类别中选择最合适的一个：[{topics}]。只返回类别名称，不要标点符号。\n\n摘要：{first_page_text}"
    
    try:
        category = ai.chat_with_gemini(prompt).strip()
        category = category.replace("'", "").replace('"', "").replace(".", "")
        print(f"分类结果: {category}")
        
        move_file_to_category(file_path, category)
        print(f"处理完成。")
    except Exception as e:
        print(f"[警告] 分类或移动失败，但已入库: {e}")

def process_image(ai, db, file_path):
    """处理单张图片的逻辑"""
    filename = os.path.basename(file_path)
    print(f"\n[IMG] 正在处理: {filename}")
    
    # 去重，检查数据库中是否已存在
    if db.check_image_exists(filename):
        print(f"[跳过]: 数据库中已存在该图片")
        return

    # 1. CLIP 向量
    try:
        clip_vec = ai.get_clip_embedding(file_path)
    except Exception as e:
        print(f"[跳过]: CLIP处理失败 {e}")
        return

    # 2. Gemini 描述
    print("Gemini 正在观察图片...")
    try:
        desc = ai.get_image_description(file_path)
        print(f"描述: {desc[:30]}...")
    except Exception as e:
        print(f"[跳过]: Gemini描述生成失败 {e}")
        return
    
    # 3. 描述向量化
    gemini_vec = ai.get_gemini_embedding(desc)

    # 4. 入库
    db.add_image(file_path, clip_vec, desc, gemini_vec)
    print("图片已双路入库。")


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
            process_image(ai, db, args.path)
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
                    process_image(ai, db, full_path)
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

        print("\n参考片段")
        for i, meta in enumerate(metas):
            print(f"[{i+1}] {meta.get('source', '未知')} (P.{meta.get('page', '?')})")

        print("\nGemini 回答:")
        context = "\n\n".join(docs)
        prompt = f"你是一个学术助手。基于以下参考资料回答用户问题：{args.query}\n\n参考资料：\n{context}"
        answer = ai.chat_with_gemini(prompt)
        print(answer)

    # 5. 文件索引 (List Papers)
    elif args.command == "list_papers":
        print(f"正在索引主题: '{args.topic}' ...")
        
        # 1. 向量化查询
        query_vec = ai.get_gemini_embedding(args.topic)
        
        # 2. 扩大搜索范围 取前20个相关片段，以覆盖更多文件
        results = db.search_paper(query_vec, n_results=20)
        
        metas = results['metadatas'][0]
        distances = results['distances'][0]

        if not metas:
            print("[提示] 未找到相关论文。")
            return

        # 3. 聚合与去重
        found_files = {} 
        # { "filename": {"path": full_path, "score": distance, "count": 1} }

        for meta, dist in zip(metas, distances):
            path = meta.get('source', 'Unknown')
            filename = os.path.basename(path)
            
            if filename not in found_files:
                found_files[filename] = {
                    "path": path,
                    "score": dist, # 越小越相关
                    "count": 1
                }
            else:
                found_files[filename]["count"] += 1
                # 如果发现了更相关的段落，更新分数
                if dist < found_files[filename]["score"]:
                    found_files[filename]["score"] = dist

        # 4. 排序
        sorted_files = sorted(found_files.items(), key=lambda x: x[1]['score'])

        print(f"\n找到 {len(sorted_files)} 篇相关论文 (按相关度排序):\n")
        print(f"{'序号':<5} {'匹配度':<10} {'文件名'}")
        print("-" * 60)
        
        for i, (fname, info) in enumerate(sorted_files):
            relevance = max(0, (1 - info['score'])) * 100 
            print(f"[{i+1}]   {relevance:.1f}%      {fname}")

        print("-" * 60)


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
                print(f"  {i+1}. {os.path.basename(meta['path'])} | {desc}...")
        
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
        
        # 2. CLIP (fallback)
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