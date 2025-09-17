# -*- coding: utf-8 -*-

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
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

if platform.system() == "Linux" or platform.system() == "Darwin":
    # next lines for fix streamlit: Your system has an unsupported version of sqlite3.
    # Chroma requires sqlite3 >= 3.35.0 in cloud streamlit.app
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

warnings.filterwarnings('ignore')

# published on https://pse-project-rag-pure.streamlit.app/
# admin application via HitHub account  on https://share.streamlit.io/

# пути к файлам
texts_pkl = "./pickles/texts_4k_token_pkl.pkl"     # "./pickles/texts_pkl.pkl"
text_summaries_pkl = "./pickles/text_summaries_pkl.pkl"
tables_pkl = "./pickles/tables_pkl.pkl"
table_summaries_pkl = "./pickles/table_summaries_pkl.pkl"
img_base64_list_pkl = "./pickles/img_base64_list.pkl"
image_summaries_pkl = "./pickles/image_summaries_pkl.pkl"
model = "gpt-4o"   # "gpt-3.5-turbo"

# получение имения хоста и платформы для дальнейшего вывода
hostname = sckt.gethostname()
plat = platform.system()

# определение необходимых для запуска RAG-pipeline PDF файла функций
########################################################################################################################
# Функция добавления документов в ритривер
def add_document_to_retr(retriever: MultiVectorRetriever, doc_summaries, doc_contents):
    """
    Функция для добавления документов и их метаданных в ритривер.

    Аргументы:
    retriever: Ретривер, в который будут добавляться документы.
    doc_summaries: Список суммаризаций документов.
    doc_contents: Список исходных содержимых документов.
    """
    # Генерируем уникальные идентификаторы для каждого документа
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    id_key = "doc_id"  # Ключ для идентификации документов в хранилище
    # Создаем документы для векторного хранилища из суммаризаций
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    #print(summary_docs[:1])
    # Добавляем документы в векторное хранилище
    retriever.vectorstore.add_documents(summary_docs)

    # Добавляем метаданные документов в хранилище
    #print(doc_contents[:1])
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # return retriever  # Возвращаем созданный ритривер


# Функция создания многофакторного ритривера для базы данных
def create_multi_vector_retriever(vectorstore,
                                  text_summaries,
                                  texts,
                                  table_summaries,
                                  tables,
                                  image_summaries,
                                  images
                                  ):
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
    store = InMemoryStore()
    id_key = "doc_id"  # Ключ для идентификации документов в хранилище
    # Создаем многофакторный ритривер
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )
    if text_summaries:
        # print (text_summaries[:1])
        # print(texts[:1])
        add_document_to_retr(retriever, text_summaries, texts)
    if table_summaries:
        add_document_to_retr(retriever, table_summaries, tables)
    if image_summaries:
        add_document_to_retr(retriever, image_summaries, images)

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
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    print (f"len(texts)= {len(texts)}\nlen(b64_images)= {len(b64_images)}\n")
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
            "При подготовке ответа используй все предоставленные тебе данные не учитывая их формат"
            " (текст, таблица, изображение/картинка). При выявлении противоречий или инсайтов - обрати на это"
            " внимание в своем ответе/заключении. Вопрос пользователя: {data_dict['question']}\n\n"
            "Текст и / или таблицы:\n{formatted_texts}"
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

def multi_modal_worag_chain(retriever):
    """
    Создает RAG цепочку для работы с мультимодальными запросами, включая текст и изображения.

    Аргументы:
    retriever: Ритривер для получения данных.

    Возвращает:
    Цепочка для обработки запросов с учетом текста и изображений.
    """
    # OpenAI API ключ в os.environ["OPENAI_API_KEY"]
    gen_ai_model = ChatOpenAI(temperature=0, model=model, max_tokens=3000)
    # Определяем цепочку обработки запросов
    chain = (
        {
            "context": RunnableLambda(split_image_text_types),
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
# включение/выключение RAG и вывод информации о проекте
rag_mode = True
if "rag_mode" not in st.session_state:
    st.session_state["rag_mode"] = True
else:
    rag_mode = st.session_state["rag_mode"]

if "rag_mode" in st.session_state:
    rag_mode = st.checkbox("RAG", value=st.session_state["rag_mode"])
    st.session_state["rag_mode"] = rag_mode

if rag_mode:
    st.title(":red[GPT]+:green[RAG]+:blue[Streamlit]:red[=Great!]:smiley:")
else:
    st.title(":red[GPT]+:blue[Streamlit]:red[=Good]:confused:")

st.write("**Cource: :blue[LLM's - from architecture to building multimodal systems]**")
st.write("**2025.09.22 Panarin S.E. - project :green[Multimodal RAG system]**")
st.write(f"host: :blue[{hostname}] OS: :blue[{plat}] model: :red[{model}]")

if st.button("Reset dialog"):
    # clear chat history
    if "messages" in st.session_state:
        st.session_state.messages.clear()

########################################################################################################################
# при первом запуске данные считываем из pkl файлов с диска при обновлении WEB страницы - из сессионной памяти
########################################################################################################################

print(f"проверка и загрузка из сохраненных pkl файлов или из сессионной памяти")
if "texts" not in st.session_state:
    # text элементы
    if not os.path.exists(texts_pkl):
        print(f"\t\tfile not exists:{texts_pkl}")
        exit(2)
    else:
        print(f"\t\tfile loaded - ok: {texts_pkl}")
        with open(texts_pkl, 'rb') as inp:
            texts = pickle.load(inp)
            st.session_state["texts"] = texts
else:
    texts = st.session_state["texts"]
    print(f"\t\ttexts - loaded from session memory")


if "tables" not in st.session_state:
    # table элементы
    if not os.path.exists(tables_pkl):
        print(f"\t\tfile not exists:{tables_pkl}")
        exit(2)
    else:
        print(f"\t\tfile loaded - ok: {tables_pkl}")
        with open(tables_pkl, 'rb') as inp:
            tables = pickle.load(inp)
            st.session_state["tables"] = tables
else:
    tables = st.session_state["tables"]
    print(f"\t\ttables - loaded from session memory")


if "text_summaries" not in st.session_state:
    # summary text элементы
    if not os.path.exists(text_summaries_pkl):
        print(f"\t\tfile not exists:{text_summaries_pkl}")
        exit(2)
    else:
        print(f"\t\tfile loaded - ok: {text_summaries_pkl}")
        with open(text_summaries_pkl, 'rb') as inp:
            text_summaries = pickle.load(inp)
            st.session_state["text_summaries"] = text_summaries
else:
    text_summaries = st.session_state["text_summaries"]
    print(f"\t\ttext_summaries - loaded from session memory")


if "table_summaries" not in st.session_state:
    # summary table элементы
    if not os.path.exists(table_summaries_pkl):
        print(f"\t\tfile not exists:{table_summaries_pkl}")
        exit(2)
    else:
        print(f"\t\tfile loaded - ok: {table_summaries_pkl}")
        with open(table_summaries_pkl, 'rb') as inp:
            table_summaries = pickle.load(inp)
            st.session_state["table_summaries"] = table_summaries
else:
    table_summaries = st.session_state["table_summaries"]
    print(f"\t\ttable_summaries - loaded from session memory")


if "img_base64_list" not in st.session_state:
    # img_base64_list элементы
    if not os.path.exists(img_base64_list_pkl):
        print(f"\t\tfile not exists:{img_base64_list_pkl}")
        exit(2)
    else:
        print(f"\t\tfile loaded - ok: {img_base64_list_pkl}")
        with open(img_base64_list_pkl, 'rb') as inp:
            img_base64_list = pickle.load(inp)
            st.session_state["img_base64_list"] = img_base64_list
else:
    img_base64_list = st.session_state["img_base64_list"]
    print(f"\t\timg_base64_list - loaded from session memory")


if "image_summaries" not in st.session_state:
    # summary image элементы
    if not os.path.exists(image_summaries_pkl):
        print(f"\t\tfile not exists:{image_summaries_pkl}")
        exit(2)
    else:
        print(f"\t\tfile loaded - ok: {image_summaries_pkl}")
        with open(image_summaries_pkl, 'rb') as inp:
            image_summaries = pickle.load(inp)
            st.session_state["image_summaries"] = image_summaries
else:
    image_summaries = st.session_state["image_summaries"]
    print(f"\t\timage_summaries - loaded from session memory")

########################################################################################################################
# начало реального запуска RAG-pipeline
# создание или загрузка из сессионной памяти объектов векторного хранилища, ретривера и RAG цепочки
########################################################################################################################

print(f"создание или загрузка из сессионной памяти объектов векторного хранилища, ретривера и RAG цепочки" )
# для однократной инициализации vectorstore в сессии
if "vectorstore" not in st.session_state:
    # Создаем векторное хранилище для хранения векторных представлений документов
    print(f"\t\tсоздаем векторное хранилище")
    vectorstore = Chroma(
        collection_name="pse_rag_sber_report",  # Название коллекции
        embedding_function=OpenAIEmbeddings(),  # Функция для создания векторных представлений
    )
    st.session_state["vectorstore"] = vectorstore
else:
    print(f"\t\tfetch from session memory векторное хранилище")
    vectorstore = st.session_state["vectorstore"]

# для однократной инициализации retriever_multi_vector_img в сессии
if "retriever_multi_vector_img" not in st.session_state:
    # Создаем ретривер, добавляя суммаризации текстов, таблиц и изображений
    print(f"\t\tсоздаем ретривер и добавляем суммаризации текстов, таблиц и изображений")
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list
    )
    st.session_state["retriever_multi_vector_img"] = retriever_multi_vector_img
else:
    print(f"\t\tfetch from session memory ретривер и добавляем суммаризации текстов, таблиц и изображений")
    retriever_multi_vector_img = st.session_state["retriever_multi_vector_img"]

# для однократной инициализации chain_multimodal_rag в сессии
if "chain_multimodal_rag" not in st.session_state:
    # создаем RAG цепочку с использованием ретривера
    print(f"\t\tсоздаем RAG цепочку с использованием ретривера")
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
    st.session_state["chain_multimodal_rag"] = chain_multimodal_rag
else:
    print(f"\t\tfetch from session memory RAG цепочку с использованием ретривера")
    chain_multimodal_rag = st.session_state["chain_multimodal_rag"]

# для однократной инициализации chain_multimodal_worag в сессии
if "chain_multimodal_worag" not in st.session_state:
    # создаем цепочку без RAG с использованием ретривера
    print(f"\t\tсоздаем цепочку без RAG с использованием ретривера")
    chain_multimodal_worag = multi_modal_worag_chain(retriever_multi_vector_img)
    st.session_state["chain_multimodal_worag"] = chain_multimodal_worag
else:
    print(f"\t\tfetch from session memory  цепочку без RAG с использованием ретривера")
    chain_multimodal_worag = st.session_state["chain_multimodal_worag"]

########################################################################################################################
# работа с LLM с RAG либо без него
########################################################################################################################
hello = "Привет! Готов отвечать на любые вопросы - спрашивай!"
print(f"{hello}")
# системный промпт для варианта без RAG
sysp = ("Ты — эксперт и аналитик, выдающий ответ/заключение на заданный вопрос, тему. Если конкретной информации на"
        " заданный вопрос или тему нет или недостаточно, то ничего не придумывай, просто ответь, что у тебя нет"
        " информации или ее недостаточно. ")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(hello,
                           accept_file="multiple",
                           file_type=["jpg"]):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt.text})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt.text)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if rag_mode:
            resp = chain_multimodal_rag.invoke(str(st.session_state.messages))     # .invoke(str(prompt))
        else:
            resp = chain_multimodal_worag.invoke(sysp + str(st.session_state.messages)) # .invoke(sysp + str(prompt))
        print(resp)
        st.write(resp)
    st.session_state.messages.append({"role": "assistant", "content": resp})

########################################################################################################################
# тестовые вопросы для проверки RAG
########################################################################################################################
# Тестовые вопросы

# по табличным данным
# "Что говорится в отчете Сбера о кредитах по амортизированной и справедливой стоимости на конец 2022 и 2023 года."

# по текстовым данным
# "Что можно сказать о достижениях Сбера на основании его годового отчета за 2023 год?"

# по изображению
# "Что написано в заголовке обложки отчета в годовом отчете Сбера за 2023 год на странице, где изображена"
#          " женщина-велосипедист в защитном шлеме и очках на фоне размытого пейзажа и какой текст содержится, в правой"
#          " части слайда - перечисли темы.")
# ожидаем 10-10

# "Опиши, что изображено на картинках и есть ли на изображениях банковские карты с изображением кота?"
#  это карты на 2-х картинках кот на одной девушка с зелеными волосами 34-81 41-99 41-100
#  это глюк - картинка кота с телефоном не посылалась 42-109 и девушки с зелеными волосами, но на фоне цветущей сакуры
# Кто изображен на картинке?  (figure-41-102.jpg - Чебурашка)
