import gradio as gr
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

def load_split_docs():
    loader = TextLoader("documents/your_file.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

docs = load_split_docs()
vector_store = create_vector_store(docs)
retriever = vector_store.as_retriever()

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def rag_chatbot(question):
    result = qa_chain({"query": question})
    sources = "\n\n".join([doc.page_content[:200] + "..." for doc in result["source_documents"]])
    return result["result"], sources

demo = gr.Interface(
    fn=rag_chatbot,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
    title="📚 RAG Chatbot using Mistral & FAISS",
    description="Ask questions from uploaded knowledge base using open-source models."
)

if __name__ == "__main__":
    demo.launch()
