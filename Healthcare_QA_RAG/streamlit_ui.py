import streamlit as st
import streamlit.components.v1 as components
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

@st.cache_resource
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

@st.cache_resource
def create_vector_store(_docs, index_name):  # index_name 인자 추가
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(f"faiss_index/{index_name}")  # 인덱스를 PDF 이름 기반으로 저장
    return vectorstore

@st.cache_resource
def get_vectorstore(_docs, index_name):
    path = f"faiss_index/{index_name}/index.faiss"
    if os.path.exists(path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(f"faiss_index/{index_name}", embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_store(_docs, index_name)

@st.cache_resource
def initialize_rag_chain(*pdf_paths):
    all_pages = []
    for path in pdf_paths:
        pages = load_pdf(path)
        all_pages.extend(pages)  # 문서 리스트를 병합

    # PDF 파일명을 합쳐 인덱스 이름 생성 (순서 고려)
    index_name = "_".join(sorted([
    os.path.splitext(os.path.basename(path))[0] for path in pdf_paths]))

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

#####################################################################################################################

if "page" not in st.session_state:
    st.session_state["page"] = "home"

def sidebar_menu():
    with st.sidebar:
        st.markdown("""
            <style>
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 0.5rem;
                padding-right: 0.5rem;
                padding-left: 0.5rem;
                padding-bottom: 0.5rem;
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
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <h1 style='color: #27408b; margin-top: -4rem;'>메뉴</h1>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='margin: -0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

        if st.button("🏠 홈으로", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

        if st.button("💬 새 채팅", use_container_width=True):
            st.session_state["page"] = "chat"
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

        st.markdown("##### 이전 채팅", unsafe_allow_html=True)

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
        """, height=500)


        st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

        if st.button("📄 PDF 보기", use_container_width=True):
            st.session_state["page"] = "pdf_view"
            st.rerun()

        st.markdown("<hr style='margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

        login_label = "로그아웃" if st.session_state.get("logged_in", False) else "로그인"

        if "login_link_clicked" not in st.session_state:
            st.session_state["login_link_clicked"] = False

        st.markdown(f"""
            <p style="
                text-align: right;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            ">
                <a href="#" onclick="window.parent.postMessage({{ type: 'LOGIN_CLICK' }}, '*'); return false;"
                style="
                    color: #27408b;
                    text-decoration: underline;
                    font-size: 14px;
                    cursor: pointer;
                ">{login_label}</a>
            </p>
        """, unsafe_allow_html=True)

        st.markdown("""
            <script>
            window.addEventListener("message", (event) => {
                if (event.data && event.data.type === "LOGIN_CLICK") {
                    const streamlitEvent = new CustomEvent("streamlit_login_click");
                    window.dispatchEvent(streamlitEvent);
                }
            });

            window.addEventListener("streamlit_login_click", (event) => {
                Streamlit.setComponentValue("login_click_event");
            });
            </script>
        """, unsafe_allow_html=True)

        if "login_click_event" in st.session_state:
            st.session_state.pop("login_click_event")
            st.session_state["page"] = "login"
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

    pdf_path1 = "./who.pdf"
    pdf_path2 = "./Healthcare_Vocab.pdf"
    rag_chain = initialize_rag_chain(pdf_path1, pdf_path2)

    chat_history = StreamlitChatMessageHistory(key="chat_messages")

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )

    if "messages" not in st.session_state or st.session_state["messages"] == []:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "의약품 및 질병에 대해 무엇이든 물어보세요!"
        }]
        if "first_question" in st.session_state:
            first_q = st.session_state.pop("first_question")
            st.session_state["messages"].append({
                "role": "user",
                "content": first_q
            })

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    prompt_message = st.chat_input("질문을 입력하세요", key="chat_input_chat")
    if prompt_message:
        st.chat_message("human").write(prompt_message)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"session_id": "any"}}
                response = conversational_rag_chain.invoke(
                    {"input": prompt_message},
                    config
                )
                answer = response['answer']
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