import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def create_vector_store(_docs, index_name):  # index_name 인자 추가
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(f"faiss_index/{index_name}")  # 인덱스를 PDF 이름 기반으로 저장
    return vectorstore

def get_vectorstore(_docs, index_name):
    path = f"faiss_index/{index_name}/index.faiss"
    if os.path.exists(path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(f"faiss_index/{index_name}", embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_store(_docs, index_name)

def initialize_rag_chain(*pdf_paths):
    all_pages = []
    for path in pdf_paths:
        pages = load_pdf(path)
        all_pages.extend(pages)  # 문서 리스트를 병합

    # PDF 파일명을 합쳐 인덱스 이름 생성 (순서 고려)
    index_name = "_".join([
        os.path.splitext(os.path.basename(path))[0] for path in pdf_paths
    ])

    vectorstore = get_vectorstore(all_pages, index_name)
    retriever = vectorstore.as_retriever()


    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. \
    대답은 한국어로 하고, 존댓말을 써주세요.\

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
