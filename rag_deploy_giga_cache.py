# -*- coding: utf-8 -*-

import sys
import os
import pickle
import io
import re
import base64
import uuid
import warnings
import platform
import socket as sckt
from PIL import Image
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import logging
from giga_util import get_giga_credentials, get_giga_url_access_mode, get_giga_token_access
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain.storage._lc_store import create_kv_docstore

if platform.system() == "Linux": # or platform.system() == "Darwin"
    # next lines for fix streamlit: Your system has an unsupported version of sqlite3.
    # Chroma requires sqlite3 >= 3.35.0 in cloud streamlit.app
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# set logging level - for logging to file add: filename='myapp.log',
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='\t\t%(asctime)s - %(levelname)s - %(message)s')

if not sys.warnoptions:
    warnings.simplefilter("ignore") # default Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "ignore" # ignore Also affect subprocesses

warnings.filterwarnings('ignore', category=DeprecationWarning)

# published on https://pse-project-rag-pure.streamlit.app/
# admin application via HitHub account  on https://share.streamlit.io/
giga = True
model_giga = "GigaChat-2-Pro" # "GigaChat-2-Pro" "GigaChat-2-Max"
model_emb = "Embeddings" # EmbeddingsGigaR
if giga:
    credentials = get_giga_credentials()
    if credentials == '':
        logger.critical('OS variable: GIGACHAT_CREDENTIALS not set')
        exit(1)
    # get url_oauth and access_mode
    url_oauth, access_mode = get_giga_url_access_mode()
    rc, tk = get_giga_token_access(url_oauth, credentials)
    if rc:
        payload = {}
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {tk}'
        }
    else:
        logger.critical('Can''t get authorization token to GigaChat')
        exit(1)

# пути к файлам
print(f"setting up output file paths")
if giga:
    path_to_pkl = "./giga_pickles"
    path_to_db = "./db_vector"
    path_to_ds = "./db_store"
else:
    path_to_pkl = "./pickles"

# пути к файлам

all_vector_docs_pkl = os.path.join(path_to_pkl,"all_vector_docs_pkl.pkl")
print(f"\t\t\t{all_vector_docs_pkl}")

all_id_docs_pkl = os.path.join(path_to_pkl,"all_id_docs_pkl.pkl")
print(f"\t\t\t{all_id_docs_pkl}")

model = "gpt-4o"   # "gpt-3.5-turbo"

# получение имения хоста и платформы для дальнейшего вывода
hostname = sckt.gethostname()
plat = platform.system()

# определение необходимых для запуска RAG-pipeline PDF файла функций
########################################################################################################################

# Функция добавления документов в ритривер
def add_document(doc_summaries):
    """
    Функция для добавления документов и их метаданных в ритривер.

    Аргументы:
    retriever: Ретривер, в который будут добавляться документы.
    doc_summaries: Список суммаризаций документов.
    doc_contents: Список исходных содержимых документов.
    """
    # Генерируем уникальные идентификаторы для каждого документа
    doc_ids = [str(uuid.uuid4()) for _ in doc_summaries]
    id_key = "doc_id"  # Ключ для идентификации документов в хранилище
    source_document = 'source_document'
    page_number = 'page_number'
    paragraph_number = 'paragraph_number'
    text = 'text'
    table_content = 'table_content'
    image_content = 'image_content'
    image_path = 'image_path'

    # Создаем документы для векторного хранилища из суммаризаций
    # summary_docs = [
    #     Document(page_content=str(s.get('text')), metadata={id_key: doc_ids[i], source_document: s.get('source_document')})
    #     for i, s in enumerate(doc_summaries)
    # ]
    summary_docs = []
    m_d = {}
    for i, s in enumerate(doc_summaries):
        if text in s:
            p_c = str(s.get('text'))
        if table_content in s:
            p_c = str(s.get('table_content'))
        if image_content in s:
            p_c = str(s.get('image_content'))
        m_d.update({id_key: doc_ids[i]})
        if source_document in s:
            m_d.update({source_document: str(s.get(source_document))})
        if page_number in s:
            m_d.update({page_number: str(s.get(page_number))})
        if paragraph_number in s:
            m_d.update({paragraph_number: str(s.get(paragraph_number))})
        if image_path in s:
            m_d.update({image_path: str(s.get(image_path))})
        doc = Document(page_content=p_c, metadata=m_d)
        summary_docs.append(doc)

    print(f'\t\tsummary_docs = [{summary_docs[:1]}]')
    return doc_ids, summary_docs


# Функция создания многофакторного ритривера для базы данных
def create_new_multi_vector_retriever(vectorstore, all_docs_store):
    """
    Функция для создания ретривера, который может извлекать данные из разных источников (тексты, таблицы, изображения).

    Аргументы:
    vectorstore: Векторное хранилище для хранения векторных представлений документов.
    text_summaries: Список суммаризаций текстовых элементов.
    texts: Список исходных текстов.
    table_summaries: Список суммаризаций таблиц.
    tables: Список исходных таблиц.
    image_summaries: Список суммаризаций изображений.
    images: Список изображений в формате base64.

    Возвращает:
    Созданный ретривер, который может извлекать данные из различных источников.
    """

    # Создаем хранилище для метаданных документов в памяти
    # store = all_docs_store # InMemoryStore()
    id_key = "doc_id"  # Ключ для идентификации документов в хранилище
    # Создаем многофакторный ритривер
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=all_docs_store,
        id_key=id_key
    )

    return retriever

def looks_like_base64(sb):
    """
    Проверяет, выглядит ли строка как base64.

    Аргументы:
    sb: Строка для проверки.

    Возвращает:
    True, если строка выглядит как base64, иначе False.
    """
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Проверяет, является ли base64 данные изображением, проверяя сигнатуры данных.

    Аргументы:
    b64data: Строка base64, представляющая изображение.

    Возвращает:
    True, если данные начинаются с сигнатуры изображения, иначе False.
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Изменяет размер изображения, закодированного в формате base64.

    Аргументы:
    base64_string: Строка base64, представляющая изображение.
    size: Новый размер изображения.

    Возвращает:
    Закодированное в формате base64 изображение нового размера.
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Изменение размера изображения с использованием алгоритма LANCZOS для улучшения качества
    resized_img = img.resize(size, Image.LANCZOS)

    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Разделяет документы на изображения и текстовые данные.

    Аргументы:
    docs: Список документов, содержащих изображения (в формате base64) и текст.

    Возвращает:
    Словарь с двумя списками: изображения и тексты.
    """
    b64_images = []
    texts = []

    for dc in docs:
        doc = dc.decode()
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    print (f"\t\tlen(docs) = {len(docs)} len(texts) = {len(texts)} len(b64_images) = {len(b64_images)}\n")
    for d in texts:
        print(f'\t\trd = [{d}]')

    return {"images": b64_images, "texts": texts}


# Функция формирования запроса для модели с учетом изображений и текста
def img_prompt_func(data_dict):
    """
    Формирует запрос к модели с учетом изображений и текста.

    Аргументы:
    data_dict: Словарь, содержащий тексты и изображения, а также вопрос пользователя.

    Возвращает:
    Список сообщений для отправки модели.
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Добавляем изображения в сообщения, если они присутствуют
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Формируем текстовое сообщение с вопросом пользователя и текстовыми данными
    text_message = {
        "type": "text",
        "text": (
            # Ваш код здесь
            "Ты — эксперт и аналитик, выдающий ответ/заключение по предоставленной тебе в запросе информации. "
            "Используй все предоставленные тебе данные вне зависимости от их формата, но не додумывай свои.\n\n"
            # "При подготовке ответа используй все предоставленные тебе данные не учитывая их формат"
            #" (текст, таблица, изображение/картинка)."
            " При выявлении противоречий или инсайтов - обрати на это внимание в своем ответе/заключении. "
            f"Вопрос пользователя: {data_dict['question']}\n\n"
            "Текст и / или таблицы:\n"
            f"{formatted_texts}\n\n"
            # " Вопрос пользователя: {data_dict['question']}\n\n"
            # "Текст и / или таблицы:\n{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    Создает RAG цепочку для работы с мультимодальными запросами, включая текст и изображения.

    Аргументы:
    retriever: Ритривер для получения данных.

    Возвращает:
    Цепочка для обработки запросов с учетом текста и изображений.
    """
    # OpenAI API ключ в os.environ["OPENAI_API_KEY"]
    if giga:
        # Авторизация в сервисе GigaChat
        gen_ai_model = GigaChat(model=model_giga,
                        credentials=credentials,
                        verify_ssl_certs=False,
                        scope="GIGACHAT_API_CORP",
                        auth_url=url_oauth,
                        temperature=0,
                        profanity_check=False)
    else:
        gen_ai_model = ChatOpenAI(temperature=0, model=model, max_tokens=3000)
    # Определяем цепочку обработки запросов
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | gen_ai_model
        | StrOutputParser()
    )

    return chain


########################################################################################################################
# точка входа - начало отрисовки WEB-морды
########################################################################################################################
print(f"начало отрисовки WEB-морды")
# # включение/выключение RAG и вывод информации о проекте
# rag_mode = True
# if "rag_mode" not in st.session_state:
#     st.session_state["rag_mode"] = True
# else:
#     rag_mode = st.session_state["rag_mode"]
#
# if "rag_mode" in st.session_state:
#     rag_mode = st.checkbox("RAG", value=st.session_state["rag_mode"])
#     st.session_state["rag_mode"] = rag_mode
#
# if rag_mode:
#     st.title(":red[GPT]+:green[RAG]+:blue[Streamlit]:red[=Great!]:smiley:")
# else:
#     st.title(":red[GPT]+:blue[Streamlit]:red[=Good]:confused:")
#
# st.write("**Cource: :blue[LLM's - from architecture to building multimodal systems]**")
# st.write("**2025.09.22 Panarin S.E. - project :green[Multimodal RAG system]**")
# st.write(f"host: :blue[{hostname}] OS: :blue[{plat}] model: :red[{model}]")
#
# if st.button("Reset dialog"):
#     # clear chat history
#     if "messages" in st.session_state:
#         st.session_state.messages.clear()

########################################################################################################################
# при первом запуске данные считываем из pkl файлов с диска при обновлении WEB страницы - из cache_data
########################################################################################################################
# new variant

#@st.cache_data
def load_all_vector_docs():
    with open(all_vector_docs_pkl, 'rb') as inp:
        print(f"\t\tfile loaded - ok: {all_vector_docs_pkl}")
        return pickle.load(inp)

#@st.cache_data
def load_all_id_docs():
    with open(all_id_docs_pkl, 'rb') as inp:
        print(f"\t\tfile loaded - ok: {all_id_docs_pkl}")
        return pickle.load(inp)

print(f"загрузка данных хранилищ векторов и документов из сохраненных pkl файлов или из cache_data")
all_vector_docs = load_all_vector_docs()
all_id_docs = load_all_id_docs()


########################################################################################################################
# начало реального запуска RAG-pipeline
# создание или загрузка из cache_resource объектов векторного хранилища, ретривера и RAG цепочки
########################################################################################################################

if not os.path.exists(path_to_ds):
    print(f"\t\tсоздаем хранилище для документов на диске")
    all_docs_store = LocalFileStore(path_to_ds)
    all_docs_store.mset(all_id_docs)
else:
    print(f"\t\tоткрываем хранилище для документов на диске")
    all_docs_store = LocalFileStore(path_to_ds)

# all_docs_store = InMemoryStore()
# all_docs_store.mset(list(zip(all_indexes, all_docs)))

# Get all keys
# all_keys = list(all_docs_store.yield_keys())
# values = all_docs_store.mget(all_keys)
# print(f'\t\tall_keys={len(all_keys)} values={len(values)}')
# for i in range(5):
#     print(f"\t\t\t{all_keys[i]}: {values[i].decode()}")

#@st.cache_resource
def create_vectorstore(all_vector_docs):
    embeddings = GigaChatEmbeddings(
                model=model_emb,
                credentials=credentials,
                auth_url=url_oauth,
                scope="GIGACHAT_API_CORP",
                verify_ssl_certs=False,
            )
    if not os.path.exists(path_to_db):
        print(f"\t\tсоздаем из массива суммаризированных документов векторное хранилище: {path_to_db}")
        vs = FAISS.from_documents(
            documents=all_vector_docs,
            embedding=embeddings
        )
        vs.save_local(folder_path=path_to_db, index_name="faiss_index")
    else:
        print(f"\t\tоткрываем существующее векторное хранилище: {path_to_db}")
        vs = FAISS.load_local(folder_path=path_to_db, index_name="faiss_index",embeddings=embeddings,
                              allow_dangerous_deserialization=True)
    return vs

#@st.cache_resource
def create_retriever_multi_vector_img(vectorstore, all_docs_store):
    print(f"\t\tсоздаем ретривер и добавляем суммаризации текстов, таблиц и изображений")
    return create_new_multi_vector_retriever(vectorstore, all_docs_store)

#@st.cache_resource
def create_chain_multimodal_rag():
    print(f"\t\tсоздаем RAG цепочку с использованием ретривера")
    return multi_modal_rag_chain(retriever_multi_vector_img)


print(f"создание или загрузка из cache_resource объектов векторного хранилища, ретривера и RAG цепочки" )
vectorstore = create_vectorstore(all_vector_docs)
retriever_multi_vector_img = create_retriever_multi_vector_img(vectorstore, all_docs_store)
chain_multimodal_rag = create_chain_multimodal_rag()

# Пример запроса
query = ("Каким учащимся подготовлен реферат об исследовании создания первых электрических элементов и альтернативных"
         " источников энергии и каковы основные тезисы реферата?")
query = "Что говорится в отчете Сбера о кредитах по амортизированной и справедливой стоимости на конец 2022 и 2023 года?"
query = ("О чем страница отчета Сбера, где изображена женщина-велосипедист в защитном шлеме и очках на фоне размытого пейзажа?"
         " Перечисли основные темы. Существуют ли на странице оформительские ошибки и если есть, то опиши их суть.")
docs = retriever_multi_vector_img.get_relevant_documents(query, limit=6)
print(f'\t\tget_relevant_documents len(docs) = {len(docs)})')

# for d in docs:
#     print(f'\t\trd = [{d.decode()}]')
resp = chain_multimodal_rag.invoke(query)
print(f'\n\t\tresp = [{resp}]')

########################################################################################################################
# работа с LLM с RAG
########################################################################################################################
# hello = "Привет! Готов отвечать на любые вопросы - спрашивай!"
# print(f"{hello}")
# # системный промпт для варианта без RAG
# sysp = ("Ты — эксперт и аналитик, выдающий ответ/заключение на заданный вопрос, тему. Если конкретной информации на"
#         " заданный вопрос или тему нет или недостаточно, то ничего не придумывай, просто ответь, что у тебя нет"
#         " информации или ее недостаточно. ")
#
# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# # Accept user input
# if prompt := st.chat_input(hello,
#                            accept_file="multiple",
#                            file_type=["jpg"]):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt.text})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt.text)
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         if rag_mode:
#             resp = chain_multimodal_rag.invoke(str(st.session_state.messages))     # .invoke(str(prompt))
#         else:
#             resp = chain_multimodal_worag.invoke(sysp + str(st.session_state.messages)) # .invoke(sysp + str(prompt))
#         print(resp)
#         st.write(resp)
#     st.session_state.messages.append({"role": "assistant", "content": resp})

########################################################################################################################
# тестовые вопросы для проверки RAG
########################################################################################################################
# Тестовые вопросы

# по табличным данным
# "Что говорится в отчете Сбера о кредитах по амортизированной и справедливой стоимости на конец 2022 и 2023 года?"

# по текстовым данным
# "Что говорится о главных достижениях Сбера в его годовом отчете за 2023 год?"

# по изображению
# "Что написано в заголовке обложки отчета в годовом отчете Сбера за 2023 год на странице, где изображена"
#          " женщина-велосипедист в защитном шлеме и очках на фоне размытого пейзажа и какой текст содержится, в правой"
#          " части слайда - перечисли темы.")
# ожидаем 10-10
#  О чем страница отчета Сбера, где изображена женщина-велосипедист в защитном шлеме и очках на фоне размытого пейзажа?"
# "Опиши, что изображено на картинках и есть ли на изображениях банковские карты с изображением кота?"
#  это карты на 2-х картинках кот на одной девушка с зелеными волосами 34-81 41-99 41-100
#  это глюк - картинка кота с телефоном не посылалась 42-109 и девушки с зелеными волосами, но на фоне цветущей сакуры
# Что и ли кто изображен на картинке?  (figure-41-102.jpg - Чебурашка)
