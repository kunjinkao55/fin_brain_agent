from openai import OpenAI
import chromadb

client = OpenAI()
chroma = chromadb.Client()
collection = chroma.create_collection("knowledge_base")

def add_documents(docs: list[str]):
    """将文档添加到知识库"""
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=docs
    ).data
    
    collection.add(
        documents=docs,
        embeddings=[e.embedding for e in embeddings],
        ids=[f"doc_{i}" for i in range(len(docs))]
    )

def rag_query(question: str, top_k: int = 3) -> str:
    """RAG 问答"""
    # 检索相关文档
    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    ).data[0].embedding
    
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k
    )
    
    context = "\n".join(results['documents'][0])
    
    # 生成答案
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"基于以下上下文回答问题：\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content