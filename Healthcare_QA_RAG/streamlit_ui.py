import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model="gpt-4o-mini")

def create_translation_chain(llm):
    translation_prompt = PromptTemplate.from_template("""
아래 한국어 의학 질문을 영어로 번역해주세요.

다음 규칙을 반드시 따르세요:

1. **의학 용어**는 WHO 또는 국제적으로 공인된 **표준 의학 용어**로 번역할 것
-> **일반적으로 쓰이는 용어가 아닌 전문적인 의학 용어로 번역하시오.**
2. **공식 영어 용어가 2가지 이상 예측될 경우**, 가능성이 높은 순으로 3개로 리스트에 넣어 나열할 것 (예: "[ liver cirrhosis , hepatic cirrhosis ]")
3. **공식 약어가 있는 경우**, 전체 용어를 먼저 쓰고 괄호 안에 약어를 함께 표기할 것 (예: "chronic obstructive pulmonary disease (COPD)")
4. **리스트에 넣을 단어의 개수는 3개이다.(확률 높은 순)** (예: "what is [ coryza , upper respiratory infection , cold ]?")
5. 번역 결과는 **영어 한 문장**으로 출력하며, **한국어는 포함하지 말 것**

질문:
{input}
""")
    return LLMChain(llm=llm, prompt=translation_prompt)

@st.cache_resource
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

@st.cache_resource
def get_vectorstore(_docs):
    if os.path.exists("faiss_index/index.faiss"):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_store(_docs)
    
@st.cache_resource
def initialize_rag_chain(pdf_path):
    pages = load_pdf(pdf_path)
    vectorstore = get_vectorstore(pages)
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
    
    qa_system_prompt = """당신은 의료 분야에 특화된 질문 응답 도우미입니다.

다음에 주어진 문서 기반 정보를 사용하여 사용자의 질문에 답변해주세요. 문서에 정보가 없는 경우, 모른다고 정중히 말해주세요. 추측하거나 문서에 없는 내용을 만들어내지 마세요.
---------------
[답변 지침]
1. 질문에 대한 **정확한 정의, 정답 여부, 핵심 개념 등 대답**을 먼저 간략히 말하시오.
2. ** 핵심 **
    - 핵심 개념의 주요 특징이나 작용 방식을 설명해

3. 가능하면 아래처럼 항목을 나눠 설명해. 단, **문서에 기반한 정보만 사용**하고, 없는 정보는 절대 추론하지 말아라.
    - ① 주요 종류 또는 분류
    - ② 약물 예시 및 기전
    - ③ 적응증 및 사용 목적
    - ④ 부작용, 주의사항, 금기사항 등
    -> **문서 내용을 기반으로 사실에 근거한 구체적이고 신중한 설명**을 단락을 나눠서 작성해주세요.
    -> **해당 관련 문서에 있는 내용을 최대한 자세하게 관련된 모든 내용을 깔끔한 형식으로 출력해주세요.**
    -> 문장을 길게 쓰지 말고 보기 쉽게 풀어 쓰세요. (예: "- 효과가 나타나기까지는 수 주에서 수 개월이 걸릴 수 있음")
    -> 어려운 의학 용어는 괄호 안에 풀어쓰며, 가능한 WHO 용어를 그대로 사용해주세요.

4. WHO 문서에 없는 내용은 "자료에 따르면 제공되지 않음"이라고 명확히 밝혀라.
5. 마지막에는 간단한 마무리 멘트를 포함해라. (예: “이상으로 설명을 마칩니다.”)
---------------
[출력 형식 예시]

✅ **핵심 답변 문장**

📌 **개념 및 특징 설명**

1️⃣ **내용 분류 1**  
- 내용

2️⃣ **내용 분류 2**  
- 내용

3️⃣ **내용 분류 3**  
- 내용

ℹ️ **추가 주의사항/정보**
- 내용

✔️ **마무리 멘트**
- 내용
-----------------------
규칙:
- 마크다운 문법을 정확히 사용할 것 (`**굵은 글씨**`, `- 리스트` 등)
- **한국어로**, **존댓말**을 사용하여 정중하게 대답하세요.
- 단, 의학 용어의 경우, 한국어로 답하되, 괄호를 치고 그 안에 대응되는 영어 의료 단어를 적으시오. ex) 모르핀(morphine)
- 예외적으로 의학 용어가 한국에서 일반적으로 영어로 표기되는 경우 이는 영어(한국어) 순으로 적으시오. ex) DMARDs(질병 수정 항류머티즘 약물) 
- 문서에 근거가 없는 정보는 절대 상상하거나 생성하지 마세요.

{context}
"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

#####################################################################################################################

if "page" not in st.session_state:
    st.session_state["page"] = "home"

def sidebar_menu():
    with st.sidebar:
        st.markdown("""
            <style>
            [data-testid="stSidebar"] > div:first-child {
                display: flex;
                flex-direction: column;
                height: 100%;
                padding: 0.5rem;
            }
            div.stButton > button {
                font-size: 16px;
                padding: 0.5rem 0.5rem;
                border-radius: 8px;
                border: 1px solid #cccccc;
                background-color: #f5f5f5;
                color: #333333;
            }
            div.stButton > button:hover {
                background-color: #e0e0e0;
                border-color: #999999;
            }
            .chat-list-item {
                padding: 0.5rem 0.75rem;
                border-radius: 8px;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 4px;
                transition: background-color 0.2s ease;
            }
            .chat-list-item:hover {
                background-color: #f0f0f5;
            }
            .chat-more-button {
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
                color: #555;
            }
            .chat-context-menu {
                position: absolute;
                background: white;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                z-index: 1000;
                padding: 0.5rem;
                display: none;
            }
            .chat-context-menu.show {
                display: block;
            }
            .chat-context-menu-item {
                padding: 0.4rem 0.8rem;
                cursor: pointer;
                transition: background-color 0.2s ease;
            }
            .chat-context-menu-item:hover {
                background-color: #f5f5f5;
            }
            .chat-list-container {
                flex-grow: 1;  /* 유동적으로 늘어남 */
                overflow-y: auto;
                margin-bottom: 1rem;
            }
            .sidebar-bottom {
                margin-top: auto;
                text-align: right;
            }
            </style>
        """, unsafe_allow_html=True)

        # -------------- 상단 메뉴 ----------------
        st.markdown("""
            <h1 style='color: #27408b; margin-top: -4rem;'>메뉴</h1>
            <hr style='margin: -0.5rem 0 1.5rem 0;'>
        """, unsafe_allow_html=True)

        if st.button("🏠 홈으로", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

        if st.button("💬 새 채팅", use_container_width=True):
            st.session_state["page"] = "chat"
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

        st.markdown("##### 이전 채팅", unsafe_allow_html=True)

        # -------------- 채팅 리스트 ----------------
        st.markdown("<div class='chat-list-container'>", unsafe_allow_html=True)

        components.html("""
        <div id="chat_list_container"></div>
        <script>
        function loadChatList() {
            const chats = JSON.parse(localStorage.getItem("chat_list") || "[]");
            const chatListDiv = document.getElementById("chat_list_container");
            chatListDiv.innerHTML = "";

            if (chats.length === 0) {
                chatListDiv.innerHTML = "<p style='color: #999; font-size: 14px; text-align: center; margin-top: 0.5rem;'>저장된 채팅이 없습니다.</p>";
                return;
            }

            chats.forEach((chat, i) => {
                const chat_id = chat.id;
                const chat_name = chat.name;
                const menu_id = "menu_" + chat_id;

                const chatItem = document.createElement("div");
                chatItem.className = "chat-list-item";
                chatItem.onclick = () => window.parent.postMessage({ type: 'SELECT_CHAT', chat_id: chat_id }, '*');

                const chatNameSpan = document.createElement("span");
                chatNameSpan.textContent = chat_name;

                const moreButton = document.createElement("button");
                moreButton.className = "chat-more-button";
                moreButton.textContent = "⋯";
                moreButton.onclick = (event) => {
                    event.stopPropagation();
                    toggleContextMenu(menu_id);
                };

                chatItem.appendChild(chatNameSpan);
                chatItem.appendChild(moreButton);

                const contextMenu = document.createElement("div");
                contextMenu.id = menu_id;
                contextMenu.className = "chat-context-menu";
                contextMenu.innerHTML = `
                    <div class="chat-context-menu-item" onclick="window.parent.postMessage({ type: 'RENAME_CHAT', chat_id: '${chat_id}' }, '*')">이름 바꾸기</div>
                    <div class="chat-context-menu-item" onclick="window.parent.postMessage({ type: 'DELETE_CHAT', chat_id: '${chat_id}' }, '*')">삭제</div>
                `;
                chatListDiv.appendChild(chatItem);
                chatListDiv.appendChild(contextMenu);
            });
        }

        function toggleContextMenu(id) {
            var menus = document.querySelectorAll('.chat-context-menu');
            menus.forEach(menu => {
                if (menu.id === id) {
                    menu.classList.toggle('show');
                } else {
                    menu.classList.remove('show');
                }
            });
        }

        document.addEventListener('click', function(event) {
            if (!event.target.matches('.chat-more-button')) {
                var menus = document.querySelectorAll('.chat-context-menu');
                menus.forEach(menu => {
                    menu.classList.remove('show');
                });
            }
        });

        setTimeout(loadChatList, 100);
        </script>
        """, height=150)  # 최소 높이만 잡아줌

        st.markdown("</div>", unsafe_allow_html=True)  # chat-list-container 끝

        # -------------- 하단 버튼 ----------------
        st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

        if st.button("📄 PDF 보기", use_container_width=True):
            st.session_state["page"] = "pdf_view"
            st.rerun()
    
def show_home():
    st.markdown("""
        <div style='text-align: center; margin-top: 60px; margin-bottom: 8px;'>
            <img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png" width="150"><br><br>
            <h1 style='color: #27408b; margin-bottom: 2px;'>🩺 의약품 및 질병 Q&A 챗봇 💬</h1>
            <p style='color: #555; margin-top: 2px; margin-bottom: 2px;'>증상이나 약품명을 입력하고 건강 정보를 손쉽게 확인해보세요.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)

    first_question = st.chat_input("첫 대화를 입력해보세요", key="chat_input_home")
    if first_question:
        st.session_state["page"] = "chat"
        st.session_state["first_question"] = first_question
        st.rerun()
    

def show_chat():
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    st.header("🩺 의약품 및 질병 Q&A 챗봇 💬")

    st.markdown("""
        <style>
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 1rem;
            left: 16rem;
            right: 1rem;
            background-color: #f9f9f9;
            padding: 0.5rem;
            border-radius: 10px;
            box-shadow: 0px -2px 5px rgba(0,0,0,0.1);
            z-index: 100;
        }
        </style>
    """, unsafe_allow_html=True)

    pdf_path = "./who.pdf"
    rag_chain = initialize_rag_chain(pdf_path)

    chat_history = StreamlitChatMessageHistory(key="chat_messages")

    
    if "messages" not in st.session_state or st.session_state["messages"] == []:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "의약품 및 질병에 대해 무엇이든 물어보세요!"
        }]
        if "first_question" in st.session_state:
            prompt_message = st.session_state.pop("first_question")
            if prompt_message:
                # 0. 대화 기록에 "한국어 질문" 수동 저장
                chat_history.add_user_message(prompt_message)       # 한국어 질문
                st.chat_message("human").write(prompt_message)

            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    config = {"configurable": {"session_id": "any"}}
                    # 1. 한국어 질문 → 영어 번역
                    translation_chain = create_translation_chain(llm)
                    translated = translation_chain.invoke({"input": prompt_message})
                    translated_input = translated['text']  # 영어 질문

                    # 2. 영어 질문 → 문서 기반 QA 수행 (답변은 한국어로 생성됨)
                    response = rag_chain.invoke({"input": translated_input, "history": [] })
                    answer = response['answer']
                    chat_history.add_ai_message(response["answer"])     # 한국어 응답
                    st.write(answer)

                    with st.expander("참고 문서 확인"):
                        for doc in response['context']:
                            preview = doc.page_content.strip().replace("\n", " ")[:500]
                            source = doc.metadata.get("display_source", doc.metadata.get("source", "알 수 없음"))
                            st.markdown(f"📄 **{source}**\n\n{preview}...")

    for msg in chat_history.messages[2:]:
        st.chat_message(msg.type).write(msg.content)
    
    prompt_message = st.chat_input("질문을 입력하세요", key="chat_input_chat")
    if prompt_message:
        # 0. 기록
        chat_history.add_user_message(prompt_message)
        st.chat_message("human").write(prompt_message)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"session_id": "any"}}
                # 1. 한국어 질문 → 영어 번역
                translation_chain = create_translation_chain(llm)
                translated = translation_chain.invoke({"input": prompt_message})
                translated_input = translated['text']  # 영어 질문

                # 2. 영어 질문 → 문서 기반 QA 수행 (답변은 한국어로 생성됨)
                response = rag_chain.invoke({"input": translated_input, "history": [] })
                answer = response['answer']
                chat_history.add_ai_message(response["answer"])     # 한국어 응답
                st.write(answer)

                with st.expander("참고 문서 확인"):
                    for doc in response['context']:
                        preview = doc.page_content.strip().replace("\n", " ")[:500]
                        source = doc.metadata.get("display_source", doc.metadata.get("source", "알 수 없음"))
                        st.markdown(f"📄 **{source}**\n\n{preview}...")

def show_pdf_view():
    st.header("📄 PDF 보기")
    st.markdown("<hr>", unsafe_allow_html=True)
    reference_links = [
        {
            "name": "WHO model formulary 2008",
            "url": "https://iris.who.int/handle/10665/44053/"
        },
        {
            "name": "우리말 의학 용어 기본 원칙 pdf",
            "url": "https://www.kamje.or.kr/func/download_file?file_name=035962dc9b5c16e6b617ba0c1f076628.pdf&file_path=../uploads/board/bo_workshop/38/&orig_name=5_%EC%9A%B0%EB%A6%AC%EB%A7%90_%EC%9D%98%ED%95%99%EC%9A%A9%EC%96%B4%EC%9D%98_%EA%B8%B0%EB%B3%B8%EC%9B%90%EC%B9%99,%ED%95%84%EC%88%98%EC%9D%98%ED%95%99%EC%9A%A9%EC%96%B4%EC%A7%91%EC%9D%84_%EC%A4%91%EC%8B%AC%EC%9C%BC%EB%A1%9C1.pdf"
        },
        {
            "name": "데이터셋 by 깃허브",
            "url": "https://github.com/UsernameisKoo/RAG_Service/"
        },
    ]

    for ref in reference_links:
        st.markdown(f"<h2 style='font-size:1.5rem;'>🔗 <a href='{ref['url']}' target='_blank'>{ref['name']}</a></h2>", unsafe_allow_html=True)


def show_login():
    st.header("🔐 로그인 페이지")
    username = st.text_input("아이디")
    password = st.text_input("비밀번호", type="password")

    if st.button("로그인"):
        if username == "admin" and password == "password":
            st.success("로그인 성공!")
            st.session_state["logged_in"] = True
            st.session_state["page"] = "home"
            st.rerun()
        else:
            st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

sidebar_menu()

page_placeholder = st.empty()

if st.session_state["page"] == "home":
    with page_placeholder.container():
        show_home()
elif st.session_state["page"] == "chat":
    with page_placeholder.container():
        show_chat()
elif st.session_state["page"] == "pdf_view":
    with page_placeholder.container():
        show_pdf_view()
elif st.session_state["page"] == "login":
    with page_placeholder.container():
        show_login()