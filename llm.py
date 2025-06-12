import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


##환경변수 읽어오기 ======================================
load_dotenv()

## llm 함수 정의
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


## database 함수 정의 =========================
def get_database(index_name = 'laws'):
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    ## 임베딩 모델 지정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    Pinecone(api_key=PINECONE_API_KEY)

    #저장된 인덱스 가져오기
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )


### Statefully manage chat history ===============================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

## retrievalQA 함수 정의 
def get_retrievalQA():
    LANCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    ## LLM 모델 지정
    llm  = get_llm()

    ##vector store에서 index 정보
    database = get_database()
    retriever = database.as_retriever(search_kwargs={'k': 2})

    ## 코드추가 ######################################################
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder

    contextualize_q_system_prompt = (
        "채팅 기록과 최신 사용자 질문을 고려하여"
        "채팅 기록의 맥락을 참조할 수 있습니다."
        "이해할 수 있는 독립적인 질문을 작성하십시오."
        "채팅 기록 없이도 가능합니다. 질문에 답변하지 마십시오."
        "필요한 경우 질문을 재구성하고, 그렇지 않은 경우 그대로 반환하십시오."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
        '''[identity]

-당신은 전세 사기 피해 법률 전문가 입니다.
-[context]를 참고하여 사용자의 질문에 답변하세요.
-답변은 출처에 해당하는 ('어떤법'의 '몇조', '몇호', '몇항'의 출처) 형식으로 정확한 출처를 문단 마지막에 표시해주세요.
-항목별로 표시해서 답변해주세요.
-전세 사기 피해 법률 이외에는 '전세 사기 피해와 관련된 질문을 해주세요.'로 답하세요.
[context] 

{context} 
'''
    "\n\n"
    "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ]
    )



    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # mapped_rag_chain = rag_chain | (lambda x: {"output": x["answer"]})
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    return conversational_rag_chain


## [AI Message 함수 정의] ######################
def get_ai_message(user_message, session_id='default'):
    qa_chain = get_retrievalQA()

    ai_message = qa_chain.stream(
        {"input" : user_message},
        config={"configurable": {"session_id": session_id}},
    )

    print(f'대화이력 -> , {get_session_history(session_id)}\n\n')
    print('=' * 50 + '\n')
    
    return ai_message

