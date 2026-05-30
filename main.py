from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from rag_core import index_document, query_document
import uvicorn

app = FastAPI(title="本地 RAG 问答系统")

html_form = """
<!DOCTYPE html>
<html>
<head><title>本地知识库问答</title></head>
<body>
    <h2>上传文档（PDF）</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".pdf" required>
        <button type="submit">上传并索引</button>
    </form>

    <hr>
    <h2>提问</h2>
    <form action="/ask" method="post">
        <input type="text" name="question" style="width: 60%%;" placeholder="例如：电池续航多久？">
        <button type="submit">提问</button>
    </form>

    <hr>
    <div id="result">
        <!-- 结果会显示在这里 -->
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return html_form

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"error": "仅支持 PDF 文件"}
    content = await file.read()
    try:
        chunk_count = index_document(content, file.filename)
        return {"message": f"文档上传成功，已索引 {chunk_count} 个文本块"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask(question: str = Form(...)):
    if not question:
        return {"error": "问题不能为空"}
    answer, sources = query_document(question)
    truncated_sources = [s[:120] + "..." if len(s) > 120 else s for s in sources[:2]]
    return {
        "question": question,
        "answer": answer,
        "sources": truncated_sources
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)