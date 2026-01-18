import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

PDF_PATH = "data/pdfs"
DB_PATH = "chroma_db"
MODEL_PATH = "models/llama.gguf"

def main():
    loader = PyPDFDirectoryLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=DB_PATH
    )

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_tokens=512,
        n_ctx=2048,
        verbose=False
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    while True:
        question = input("Ask a question (type exit to quit): ")
        if question.lower() == "exit":
            break
        answer = qa.run(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
































