# Local RAG Knowledge Base Q&A System

一个完全离线的本地知识库问答系统。上传 PDF，AI 基于文档内容回答问题。

## 功能

- 完全本地运行，无需联网
- 支持 PDF 文档上传、自动索引
- 基于 RAG（检索增强生成）的智能问答
- 简洁 Web 界面

## 技术栈

- FastAPI + ChromaDB + Ollama
- 嵌入模型：nomic-embed-text
- 大模型：qwen2.5:1.5b / llama3.2:1b

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 拉取 Ollama 模型
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

# 3. 启动服务
python main.py