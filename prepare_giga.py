# -*- coding: utf-8 -*-
from accelerate.commands.config.config import description
from unstructured.documents.elements import NarrativeText, Table, Image
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import sys
import os
import pickle
import base64
import warnings
from langchain_gigachat.chat_models import GigaChat
import time
import requests
import logging
from giga_util import get_giga_credentials, get_giga_url_access_mode, get_giga_token_access
import datetime
import uuid

# set logging level - for logging to file add: filename='myapp.log',
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='\t\t%(asctime)s - %(levelname)s - %(message)s')

if not sys.warnoptions:
    warnings.simplefilter("ignore") # default Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "ignore" # ignore Also affect subprocesses

warnings.filterwarnings('ignore', category=DeprecationWarning)

# block for measure elapsed time
def date_diff_in_seconds(dt2: datetime, dt1: datetime):
    timedelta = dt2 - dt1
    return timedelta.days * 24 * 3600 + timedelta.seconds


def dhms_from_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds

########################################################################################################################
# точка входа - обработка параметров запуска
########################################################################################################################
# read and append to list run parameters
params = []
allowed_params = ['-get_raw', '-cat_txt_tbl_img', '-sum_txt_tbl', '-sum_img', '-make_vec_doc', '-doc_opt', '-get_stat']
strStart = sys.argv[0]
if len(sys.argv) > 1:
    for count, value in enumerate(sys.argv):
        if count > 0:
            params.append(value.lower())
# check run parameters
if len(params) > 0  and not set(params).issubset(allowed_params):
    print(f'Error - got unknown key(s): {list(set(params) - set(allowed_params))}\nExit script!')
    exit(10)

# debug code on alt_sources_energy.pdf work real on Sber2023.pdf
report_path = "source_pdf_report/Sber2023.pdf" #Sber2023.pdf alt_sources_energy.pdf
gl_start_datetime = datetime.datetime.now()
print(f"{gl_start_datetime.strftime('%Y.%m.%d %H:%M:%S')} ->: begin preprocessing input PDF file: {report_path}")

print(f"getting connection parameters to GigaChat")
giga = True
model_giga = "GigaChat-2-Max" # "GigaChat-2-Pro"
# GigaChat-2-Max
# GigaChat-Max
# GigaChat-Pro
# GigaChat-Plus
# GigaChat-2-Reasoning
# GigaChat-2
# GigaChat
if giga:
    max_concurrency_workers = 1
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
else:
    max_concurrency_workers = 5

# пути к файлам
print(f"setting up output file paths")
if giga:
    image_block_output_dir = "./giga_extracted_images"
    path_to_pkl = "./giga_pickles"
else:
    image_block_output_dir = "./extracted_images"
    path_to_pkl = "./pickles"
print(f"\t\tpath to the extracted image files: {image_block_output_dir}")
print(f"\t\toutput PKL files")
raw_pdf_elements_pkl = os.path.join(path_to_pkl,"raw_pdf_elements_pkl.pkl")
print(f"\t\t\t{raw_pdf_elements_pkl}")

texts_pkl = os.path.join(path_to_pkl,"texts_pkl.pkl")
print(f"\t\t\t{texts_pkl}")

txts_pkl = os.path.join(path_to_pkl,"txts_pkl.pkl")
print(f"\t\t\t{txts_pkl}")

tables_pkl = os.path.join(path_to_pkl,"tables_pkl.pkl")
print(f"\t\t\t{tables_pkl}")

tbls_pkl = os.path.join(path_to_pkl,"tbls_pkl.pkl")
print(f"\t\t\t{tbls_pkl}")

images_pkl = os.path.join(path_to_pkl,"images_pkl.pkl")
print(f"\t\t\t{images_pkl}")

text_summaries_pkl = os.path.join(path_to_pkl,"text_summaries_pkl.pkl")
print(f"\t\t\t{text_summaries_pkl}")

table_summaries_pkl = os.path.join(path_to_pkl,"table_summaries_pkl.pkl")
print(f"\t\t\t{table_summaries_pkl}")

img_base64_list_pkl = os.path.join(path_to_pkl,"img_base64_list.pkl")
print(f"\t\t\t{img_base64_list_pkl}")

image_summaries_pkl = os.path.join(path_to_pkl,"image_summaries_pkl.pkl")
print(f"\t\t\t{image_summaries_pkl}")

image_desc_base64_pkl = os.path.join(path_to_pkl,"image_desc_base64_pkl.pkl")
print(f"\t\t\t{image_desc_base64_pkl}")

imgs_pkl = os.path.join(path_to_pkl,"imgs_pkl.pkl")
print(f"\t\t\t{imgs_pkl}")

all_vector_docs_pkl = os.path.join(path_to_pkl,"all_vector_docs_pkl.pkl")
print(f"\t\t\t{all_vector_docs_pkl}")

all_id_docs_pkl = os.path.join(path_to_pkl,"all_id_docs_pkl.pkl")
print(f"\t\t\t{all_id_docs_pkl}")
########################################################################################################################
# определение необходимых для предобработки PDF файла функций
########################################################################################################################
# Функция извлечения элементов из pdf-файла
def extract_pdf_elements(fname, image_output_dir):
    """
    Функция для извлечения различных элементов из PDF-файла, таких как изображения, таблицы,
    и текста. Также осуществляется разбиение текста на части (чанки) для дальнейшей обработки.

    Аргументы:
    path: Строка, содержащая путь к директории, в которую будут сохранены извлеченные изображения.
    fname: Строка, содержащая имя PDF-файла, который необходимо обработать.

    Возвращает:
    Список объектов типа `unstructured.documents.elements`, представляющих извлеченные из PDF элементы.
    """
    return partition_pdf(
        filename=fname,                                 # Путь к файлу, который нужно обработать
        strategy="hi_res",

        # infer_table_structure=True,                     # Автоматическое определение структуры таблиц в документе
        # chunking_strategy="by_title",                   # Стратегия разбиения текста на части
        # multipage_sections=False,                       # False - разделять элементы на разных страницах на отдельные фрагменты
        # max_characters=1500,                            # Максимальное количество символов в одном чанке текста
        # new_after_n_chars=1250,                         # Число символов, после которого начинается новый чанк текста
        # combine_text_under_n_chars=250,                 # Минимальное количество символов, при котором чанки объединяются
        extract_images_in_pdf=True,                     # Указание на то, что из PDF нужно извлечь изображения
        extract_image_block_to_payload=False,           # будут ли извлеченные изображения включены в результат в виде данных (payload)
        extract_image_block_output_dir=image_output_dir,# куда будут сохраняться извлеченные изображения
        languages=["rus", "eng"]                        # языки для текста
        # unique_element_ids=True
    )

# Функция категоризации элементов
def categorize_elements(raw_pdf_elements, source_document):
    """
    Функция для категоризации извлеченных элементов из PDF-файла.
    Элементы делятся на текстовые элементы и таблицы.

    Аргументы:
    raw_pdf_elements: Список объектов типа `unstructured.documents.elements`,
                      представляющих извлеченные из PDF элементы.

    Возвращает:
    Два списка: texts (текстовые элементы) и tables (таблицы).
    """
    tables = []  # Список для хранения элементов типа "таблица"
    texts = []   # Список для хранения текстовых элементов
    text_data = []  # Список для хранения текстовых элементов с метаданными
    table_data = [] # Список для хранения элементов типа "таблица" с метаданными
    image_data = []  # Список для хранения элементов типа "image" с метаданными
    # '- '
    # Инициализация словаря для подсчета параграфов на каждой странице
    paragraph_counters = {}
    for element in raw_pdf_elements:
        # Проверка типа элемента. Если элемент является таблицей, добавляем его в список таблиц
        # if "unstructured.documents.elements.Table" in str(type(element)):
        if isinstance(element, Table):
            # tables.append(str(element))
            # Извлечение id элемента
            id_element = str(element.id)
            # Извлечение номера страницы из метаданных элемента
            page_number = element.metadata.page_number

            # Преобразование таблицы в строковое представление
            table_content = str(element)
            tables.append(str(element))

            # Добавление метаданных таблицы в список table_data
            table_data.append({
                "id_element": id_element,  # id элемента
                "source_document": source_document,  # Название или путь к исходному документу
                "page_number": page_number,          # Номер страницы, на которой находится таблица
                "table_content": table_content.replace('- ','')       # Содержимое таблицы в виде строки
            })

        # Если элемент является композитным текстовым элементом, добавляем его в список текстов
        # if "unstructured.documents.elements.CompositeElement" in str(type(element)):
        if isinstance(element, NarrativeText):
            # texts.append(str(element))
            id_element = str(element.id)
            # Извлечение номера страницы из метаданных элемента
            page_number = element.metadata.page_number
            # Если на этой странице еще нет параграфов, инициализируем счетчик параграфов
            if page_number not in paragraph_counters:
                paragraph_counters[page_number] = 1
            else:
                # Если параграфы на странице уже есть, увеличиваем счетчик
                paragraph_counters[page_number] += 1

            # Определение текущего номера параграфа на странице
            paragraph_number = paragraph_counters[page_number]

            # Извлечение текста из элемента
            text_content = str(element.text)
            texts.append(str(element.text))

            # Добавление текста и его метаданных в список text_data
            text_data.append({
                "id_element": id_element,  # id элемента
                "source_document": source_document,   # Название или путь к исходному документу
                "page_number": page_number,           # Номер страницы, на которой находится текст
                "paragraph_number": paragraph_number, # Номер параграфа на текущей странице
                "text": text_content.replace('- ','') # Сам текст
            })
        # Если элемент является image элементом, добавляем его в список images
        # if "unstructured.documents.elements.Image" in str(type(element)):
        if isinstance(element, Image):
            # Извлечение id элемента
            id_element = str(element.id)
            # Извлечение номера страницы из метаданных элемента
            page_number = element.metadata.page_number
            # Извлечение пути к изображению из метаданных элемента (если он существует)
            image_path = element.metadata.image_path if hasattr(element.metadata, 'image_path') else None

            # Добавление метаданных изображения в список image_data
            image_data.append({
                "id_element": id_element,  # id элемента
                "source_document": source_document,  # Название или путь к исходному документу
                "page_number": page_number,          # Номер страницы, на которой находится изображение
                "image_path": image_path             # Путь к изображению (если доступен)
            })

    return text_data, table_data, image_data, texts, tables # Возвращаем списки с текстами, таблицами и изображениями

# Функция для суммаризации текста и таблиц
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Функция для создания суммаризации текста и таблиц с использованием модели GPT.

    Аргументы:
    texts: Список строк (тексты), которые нужно суммировать.
    tables: Список строк (таблицы), которые нужно суммировать.
    summarize_texts: Булев флаг, указывающий, нужно ли суммировать текстовые элементы.

    Возвращает:
    Два списка: text_summaries (суммаризации текстов) и table_summaries (суммаризации таблиц).
    """

    # Шаблон для запроса к модели. Задача ассистента - создать оптимизированное описание для поиска.
    prompt_text = [
        ("system", "Ты — специалист по саммаризации - созданию кратких и содержательных резюме текста."),
        ("human", """Создай краткое, логичное и ясное по смыслу резюме из текста, следующего за ключевым словом [КОНТЕКСТ].

            Выполняй основные требования к резюме:
            - кратко выделять основные идеи, ключевые мысли;
            - избегать вывода избыточной информации и малоизвестной терминологии, жаргонных слов и аббревиатур;
            - смысл резюме должен быть понятен без исходного текста;
            - не начинай вывод резюме со слова [резюме]

            [КОНТЕКСТ]: {element}
        """),
    ]
    #- вывод резюме всегда предваряй ключевым словом [ВЫВОД].
    #             [ВЫВОД]:
    # Создаем шаблон запроса на основе строки с шаблоном
    # prompt = ChatPromptTemplate.from_template(prompt_text)
    prompt = ChatPromptTemplate(prompt_text)
    # Создаем модель для генерации суммаризаций. Устанавливаем температуру 0 для детерминированных ответов.
    if giga:
        # Авторизация в сервисе GigaChat
        model = GigaChat(model=model_giga,
                        credentials=credentials,
                        verify_ssl_certs=False,
                        scope="GIGACHAT_API_CORP",
                        auth_url=url_oauth,
                        temperature=0,
                        profanity_check=False)
    else:
        model = ChatOpenAI(temperature=0, model="gpt-4o") # OpenAI API ключ в os.environ["OPENAI_API_KEY"]


    # Определяем цепочку обработки запросов: сначала шаблон запроса, затем модель, затем парсер выходных данных
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []  # Список для хранения суммаризаций текстов
    table_summaries = []  # Список для хранения суммаризаций таблиц

    # Если есть текстовые элементы и требуется их суммирование
    if texts and summarize_texts:
        # Выполняем суммирование текстов
        #text_summaries = summarize_chain.batch(texts, config={"max_concurrency":max_concurrency_workers })
        n_files = len(texts)
        n_file = 1
        for txt in texts:
            print(f'\t\tsummarization text element {n_file} from {n_files}')
            txt.update({'text': summarize_chain.invoke(txt['text'])})
            text_summaries.append(txt)
            n_file += 1
    elif texts:
        # Если суммирование не требуется, просто передаем исходные тексты
        text_summaries = texts
    logger.debug(f'length texts = {len(texts)}\n\t\ttexts: <{texts}>\n\t\ttexts summaries: [{text_summaries}]')
    # Если есть таблицы, выполняем их суммирование
    if tables:
        # Выполняем суммирование таблиц
        # table_summaries = summarize_chain.batch(tables, config={"max_concurrency":max_concurrency_workers })
        n_files = len(tables)
        n_file = 1
        for txt in tables:
            print(f'\t\tsummarization table element {n_file} from {n_files}')
            txt.update({'table_content': summarize_chain.invoke(txt['table_content'])})
            table_summaries.append(txt)
            n_file += 1
    logger.debug(f'length tables = {len(tables)}\n\t\ttables: <{tables}>\n\t\ttables summaries: [{table_summaries}]')

    return text_summaries, table_summaries  # Возвращаем результаты суммаризации

# Функция кодирования изображения в формат base64
def encode_image(image_path):
    """
    Функция для кодирования изображения в формат base64.

    Аргументы:
    image_path: Строка, путь к изображению, которое нужно закодировать.

    Возвращает:
    Закодированное в формате base64 изображение в виде строки.
    """
    with open(image_path, "rb") as image_file:
        # Читаем файл изображения в бинарном режиме и кодируем в base64
        return base64.b64encode(image_file.read()).decode("utf-8")

# Функция для суммаризации изображения с использованием модели GPT
def image_summarize(img_base64, prompt, img_path):
    """
    Функция для получения суммаризации изображения с использованием GPT модели.

    Аргументы:
    img_base64: Строка, изображение закодированное в формате base64.
    prompt: Строка, запрос для модели GPT, содержащий инструкцию для суммаризации изображения.

    Возвращает:
    Суммаризация изображения, возвращенная моделью GPT.
    """
    # Создаем объект модели GPT с заданными параметрами
    if giga:
        # Авторизация в сервисе GigaChat
        chat = GigaChat(model=model_giga,
                        credentials=credentials,
                        verify_ssl_certs=False,
                        scope="GIGACHAT_API_CORP",
                        auth_url=url_oauth,
                        temperature=0)
        file = chat.upload_file(open(img_path, "rb"),"general")
        time.sleep(1)  # Sleep for 1 seconds
        print(f'\t\t\tuploaded file: {file.filename} got id = {file.id_}')
        # Возвращаем содержимое ответа от модели
        print(f'\t\t\tdescribing image file: {file.filename}')
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},  # Запрос для модели
                    ],
                    additional_kwargs={"attachments": [file.id_]}
                )
            ]
        )
        logger.info(f'image file: <{file.filename}> describe: [{msg.content}]')
        url = f"https://gigachat.devices.sberbank.ru/api/v1/files/{file.id_}/delete"
        response = requests.request("POST", url, headers=headers, data=payload, verify=False, cert=False)
        print(f'\t\t\tdelete file {file.filename}: {response.status_code}')
        logger.info(f'delete file {file.filename}: status_code: {response.status_code}')
        return msg.content

    else:
        chat = ChatOpenAI(model="gpt-4o", max_tokens=3000) # OpenAI API ключ в os.environ["OPENAI_API_KEY"]
        # Отправляем запрос к модели GPT
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},  # Запрос для модели
                        {
                            "type": "image_url",  # Тип содержимого - изображение
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            # Изображение в формате base64
                        },
                    ]
                )
            ]
        )
        # Возвращаем содержимое ответа от модели
        return msg.content


def generate_img_summaries(images):
    """
    Функция для генерации суммаризаций изображений из указанной директории.

    Аргументы:
    path: Строка, путь к директории с изображениями формата .jpg.

    Возвращает:
    Два списка:
    - img_base64_list: Список закодированных изображений в формате base64.
    - image_summaries: Список суммаризаций для каждого изображения.
    """
    img_base64_list = []  # Список для хранения закодированных изображений
    image_summaries = []  # Список для хранения суммаризаций изображений с метаданныим (dictionary)
    imgs = []             # Список для хранения суммаризаций изображений без метаданных
    image_desc_base64 = []  # Список для хранения суммаризаций изображений с метаданныим (dictionary) и изображений base64
    # Запрос для модели GPT
    prompt = """Ты — специалист по созданию коротких и содержательных описаний по изображениям.
        Выполняй основные требования к создаваемому описанию по изображению/картинке:
        - кратко выделяй основные образы ,идеи, ключевые мысли;
        - если на изображении содержится график или таблица, то интерпретируй их, выдели и опиши закономерности, сделай выводы;
        - если на изображении или картинке содержится лицо, человек или несколько людей, то опиши их количество, пол, возраст, одежду;
        - если на изображении содержится только фоновая картинка и мало другого графического контента, то не придумывай собственные образы, мысли или идеи, а только сделай вывод о малоинформативном контенте;
        - смысл описания должен быть понятен без рассматривания исходного изображения.
    """
    #        - избегай вывода избыточной информации и малоизвестной терминологии, жаргонных слов и аббревиатур;
    #        - не начинай вывод описания со слова [описание].

    # Если есть изображения, выполняем их описание
    if images:
        n_files = len(images)
        n_file = 1
        for image in images:
            img_path = image.get("image_path")
            if os.path.exists(img_path):
                print(f'\t\tsummarization image element {n_file} from {n_files}: {img_path}')
                base64_image = encode_image(img_path)  # Кодируем изображение в base64
                img_base64_list.append(base64_image)  # Добавляем закодированное изображение в список
                description = image_summarize(base64_image, prompt, img_path)
                image.update({'image_content': description})
                image_summaries.append(image)
                imgs.append(description)
                el_image_desc_base64 = {}
                for key, value in image.items():
                    el_image_desc_base64.update({key: value})
                el_image_desc_base64.update({'image_description': description})
                el_image_desc_base64.update({'image_base64': base64_image})
                image_desc_base64.append(el_image_desc_base64)
                # print(el_image_desc_base64)
                n_file += 1
            else:
                logger.warning(f'img_path not exist: {img_path}')
    logger.debug(f'length images = {len(images)}'
                 f'\n\t\timages: <{images}>'
                 f'\n\t\timage summaries: [{image_summaries}]'
                 f'\n\t\tlength image_desc_base64: [{len(image_desc_base64)}]')

    return img_base64_list, image_summaries, imgs, image_desc_base64  # Возвращаем результаты

# Функция создания списков уникальных идентификаторов и документов
def add_document(doc_summaries):
    """
    Функция для добавления документов и метаданных в Document.

    Аргументы:
    doc_summaries: Список суммаризаций документов.
    Возвращает:
    Два списка:
    Список doc_ids уникальных идентификаторов для каждого документа
    Список summary_docs в формате Document.
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

    # print(f'\t\tsummary_docs = [{summary_docs[:1]}]')
    return doc_ids, summary_docs

########################################################################################################################
# начало реальной обработки файла
########################################################################################################################
if  len(params) == 0 or '-get_raw' in params:
    start_datetime = datetime.datetime.now()
    print(f"{start_datetime.strftime('%Y.%m.%d %H:%M:%S')} ->: begin extract elements from PDF file: {report_path}")
    if not os.path.exists(report_path):
        print(f"file for processing not exists:{report_path}")
        exit(1)

    # Извлекаем элементы из PDF-файла с помощью функции extract_pdf_elements
    raw_pdf_elements = extract_pdf_elements(report_path, image_block_output_dir)

    # сохраняем результаты для дальнейшего использования
    with open(raw_pdf_elements_pkl, 'wb') as outp:
        pickle.dump(raw_pdf_elements, outp, pickle.HIGHEST_PROTOCOL)
    print(f'\t\telements extracted: {len(raw_pdf_elements)}')
    # end main cycle of check
    datetime_finish = datetime.datetime.now()
    delta_sec = date_diff_in_seconds(datetime_finish, start_datetime)
    el_d, el_h, el_m, el_s = dhms_from_seconds(delta_sec)
    print(f"{datetime_finish.strftime('%Y.%m.%d %H:%M:%S')} ->: end extraction in "
          f"{el_d} days {el_h} hours {el_m} min {el_s} sec")

########################################################################################################################
if  len(params) == 0 or '-cat_txt_tbl_img' in params:
    start_datetime = datetime.datetime.now()
    print(f"{start_datetime.strftime('%Y.%m.%d %H:%M:%S')} ->: begin categorization text/tables/image elements")
    # raw элементы
    if not os.path.exists(raw_pdf_elements_pkl):
        print(f"\t\tfile not exists:{raw_pdf_elements_pkl}")
        exit(2)
    else:
        # Категоризируем извлеченные элементы на текстовые и табличные с помощью функции categorize_elements
        with open(raw_pdf_elements_pkl, 'rb') as inp:
            raw_pdf_elements = pickle.load(inp)
        texts, tables, images, txts, tbls = categorize_elements(raw_pdf_elements, report_path)

        # сохраняем результаты для дальнейшего использования
        with open(texts_pkl, 'wb') as outp:
            pickle.dump(texts, outp, pickle.HIGHEST_PROTOCOL)

        with open(txts_pkl, 'wb') as outp:
            pickle.dump(txts, outp, pickle.HIGHEST_PROTOCOL)

        with open(tables_pkl, 'wb') as outp:
            pickle.dump(tables, outp, pickle.HIGHEST_PROTOCOL)

        with open(tbls_pkl, 'wb') as outp:
            pickle.dump(tbls, outp, pickle.HIGHEST_PROTOCOL)

        with open(images_pkl, 'wb') as outp:
            pickle.dump(images, outp, pickle.HIGHEST_PROTOCOL)
        print(f'\t\tcategorized elements - text: {len(texts)} table: {len(tables)} image: {len(images)}')
    datetime_finish = datetime.datetime.now()
    delta_sec = date_diff_in_seconds(datetime_finish, start_datetime)
    el_d, el_h, el_m, el_s = dhms_from_seconds(delta_sec)
    print(f"{datetime_finish.strftime('%Y.%m.%d %H:%M:%S')} ->: end categorization in "
          f"{el_d} days {el_h} hours {el_m} min {el_s} sec")

########################################################################################################################
if  len(params) == 0 or '-sum_txt_tbl' in params:
    start_datetime = datetime.datetime.now()
    print(f"{start_datetime.strftime('%Y.%m.%d %H:%M:%S')} ->: begin summarization text/table elements")
    # Вызываем функцию для суммаризации текстов и таблиц, указывая, что нужно суммировать тексты
    # text элементы
    if not os.path.exists(texts_pkl):
        print(f"\t\tfile not exists:{texts_pkl}")
        exit(2)
    else:
        # print(f"\t\tfile - ok: {texts_pkl}")
        with open(texts_pkl, 'rb') as inp:
            texts = pickle.load(inp)

    # table элементы
    if not os.path.exists(tables_pkl):
        print(f"\t\tfile not exists:{tables_pkl}")
        exit(2)
    else:
        # print(f"\t\tfile - ok: {tables_pkl}")
        with open(tables_pkl, 'rb') as inp:
            tables = pickle.load(inp)

    text_summaries, table_summaries = generate_text_summaries(texts, tables, summarize_texts=True)
    # сохраняем результаты для дальнейшего использования
    with open(text_summaries_pkl, 'wb') as outp:
        pickle.dump(text_summaries, outp, pickle.HIGHEST_PROTOCOL)

    with open(table_summaries_pkl, 'wb') as outp:
        pickle.dump(table_summaries, outp, pickle.HIGHEST_PROTOCOL)
    print(f'\t\tsummarized elements - text: {len(text_summaries)} table: {len(table_summaries)}')
    datetime_finish = datetime.datetime.now()
    delta_sec = date_diff_in_seconds(datetime_finish, start_datetime)
    el_d, el_h, el_m, el_s = dhms_from_seconds(delta_sec)
    print(f"{datetime_finish.strftime('%Y.%m.%d %H:%M:%S')} ->: end summarization text/table elements in "
          f"{el_d} days {el_h} hours {el_m} min {el_s} sec")

########################################################################################################################
if  len(params) == 0 or '-sum_img' in params:
    start_datetime = datetime.datetime.now()
    print(f"{start_datetime.strftime('%Y.%m.%d %H:%M:%S')} ->: begin summarization image elements")
    # summary image элементы
    if not os.path.exists(images_pkl):
        print(f"\t\tfile not exists:{images_pkl}")
        exit(2)
    else:
        # print(f"\t\tfile - ok: {images_pkl}")
        with open(images_pkl, 'rb') as inp:
            images = pickle.load(inp)
    # Вызываем функцию для генерации суммаризаций изображений
    img_base64_list, image_summaries, imgs, image_desc_base64 = generate_img_summaries(images)

    # сохраняем результаты для дальнейшего использования
    with open(img_base64_list_pkl, 'wb') as outp:
        pickle.dump(img_base64_list, outp, pickle.HIGHEST_PROTOCOL)
    with open(image_summaries_pkl, 'wb') as outp:
        pickle.dump(image_summaries, outp, pickle.HIGHEST_PROTOCOL)
    with open(imgs_pkl, 'wb') as outp:
        pickle.dump(imgs, outp, pickle.HIGHEST_PROTOCOL)
    with open(image_desc_base64_pkl, 'wb') as outp:
        pickle.dump(image_desc_base64, outp, pickle.HIGHEST_PROTOCOL)
    print(f'\t\tsummarized elements - image+meta: {len(image_summaries)} image w/o meta: {len(imgs)}'
          f' img_base64_list: {len(img_base64_list)} image_desc_base64: {len(image_desc_base64)}')
    datetime_finish = datetime.datetime.now()
    delta_sec = date_diff_in_seconds(datetime_finish, start_datetime)
    el_d, el_h, el_m, el_s = dhms_from_seconds(delta_sec)
    print(f"{datetime_finish.strftime('%Y.%m.%d %H:%M:%S')} ->: end summarization image elements in "
          f"{el_d} days {el_h} hours {el_m} min {el_s} sec")
    print(image_desc_base64[0])
    print(image_desc_base64[1])

########################################################################################################################
if  len(params) == 0 or '-make_vec_doc' in params or ('-make_vec_doc' in params and '-doc_opt' in params):
    start_datetime = datetime.datetime.now()
    print(f"{start_datetime.strftime('%Y.%m.%d %H:%M:%S')} ->: begin of saving vector and document data to disk")

    if '-doc_opt' in params:
        var_txt = texts_pkl
    else:
        var_txt = txts_pkl
    # txts элементы
    if not os.path.exists(var_txt):
        print(f"\t\tfile not exists:{var_txt}")
        exit(2)
    else:
        with open(var_txt, 'rb') as inp:
            txts = pickle.load(inp)

    if '-doc_opt' in params:
        var_tbl = tables_pkl
    else:
        var_tbl = tbls_pkl
    # tbls элементы
    if not os.path.exists(var_tbl):
        print(f"\t\tfile not exists:{var_tbl}")
        exit(2)
    else:
        with open(var_tbl, 'rb') as inp:
            tbls = pickle.load(inp)

    # text_summaries элементы
    if not os.path.exists(text_summaries_pkl):
        print(f"\t\tfile not exists:{text_summaries_pkl}")
        exit(2)
    else:
        with open(text_summaries_pkl, 'rb') as inp:
            text_summaries = pickle.load(inp)

    # table_summaries элементы
    if not os.path.exists(table_summaries_pkl):
        print(f"\t\tfile not exists:{table_summaries_pkl}")
        exit(2)
    else:
        with open(table_summaries_pkl, 'rb') as inp:
            table_summaries = pickle.load(inp)

    if '-doc_opt' in params:
        var_img = images_pkl
    else:
        var_img = imgs_pkl
    # imgs элементы
    if not os.path.exists(var_img):
        print(f"\t\tfile not exists:{var_img}")
        exit(2)
    else:
        with open(var_img, 'rb') as inp:
            imgs = pickle.load(inp)

    # image_summaries элементы
    if not os.path.exists(image_summaries_pkl):
        print(f"\t\tfile not exists:{image_summaries_pkl}")
        exit(2)
    else:
        with open(image_summaries_pkl, 'rb') as inp:
            image_summaries = pickle.load(inp)

    # подготавливаем векторное хранилище и хранилище документов
    all_indexes = []
    all_vector_docs = []
    all_docs = []

    print(f"\t\t\tadd text_summaries and txts...")
    id_docs, vector_docs = add_document(text_summaries)
    all_indexes.extend(id_docs)
    all_vector_docs.extend(vector_docs)
    all_docs.extend(txts)

    print(f"\t\t\tadd table_summaries and tbls...")
    id_docs, vector_docs = add_document(table_summaries)
    all_indexes.extend(id_docs)
    all_vector_docs.extend(vector_docs)
    all_docs.extend(tbls)

    print(f"\t\t\tadd image_summaries and imgs...")
    id_docs, vector_docs = add_document(image_summaries)
    all_indexes.extend(id_docs)
    all_vector_docs.extend(vector_docs)
    all_docs.extend(imgs)

    with open(all_vector_docs_pkl, 'wb') as outp:
        pickle.dump(all_vector_docs, outp, pickle.HIGHEST_PROTOCOL)

    # Create a list of docs with IDs
    print(f"\t\t\tcreate a list of docs with IDs...")
    all_id_docs = []
    for i, item in enumerate(all_docs):
        if '-doc_opt' in params:
            new_item = {}
            for key, value in item.items():
                new_item.update({key: value})
                # if isinstance(value, str):
                #     new_item.update({key: value.encode()})
            all_id_docs.append((all_indexes[i], pickle.dumps(new_item),))
        else:
            all_id_docs.append((all_indexes[i], item.encode(),))

    with open(all_id_docs_pkl, 'wb') as outp:
        pickle.dump(all_id_docs, outp, pickle.HIGHEST_PROTOCOL)

    print(f'\t\t\tall_vector_docs: {len(all_vector_docs)} all_id_docs: {len(all_id_docs)}')
    datetime_finish = datetime.datetime.now()
    delta_sec = date_diff_in_seconds(datetime_finish, start_datetime)
    el_d, el_h, el_m, el_s = dhms_from_seconds(delta_sec)
    print(f"{datetime_finish.strftime('%Y.%m.%d %H:%M:%S')} ->: end of saving vector and document data to disk"
          f"{el_d} days {el_h} hours {el_m} min {el_s} sec")

########################################################################################################################
if  len(params) == 0 or '-get_stat' in params:
    start_datetime = datetime.datetime.now()
    print(f"{start_datetime.strftime('%Y.%m.%d %H:%M:%S')} ->: begin PDF file preprocessing statistics output")
    print(f"\tchecking saved PKL files")

    # raw элементы
    if not os.path.exists(raw_pdf_elements_pkl):
        print(f"\t\tfile not exists:{raw_pdf_elements_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {raw_pdf_elements_pkl}")
        with open(raw_pdf_elements_pkl, 'rb') as inp:
            raw_pdf_elements = pickle.load(inp)

    # text элементы
    if not os.path.exists(texts_pkl):
        print(f"\t\tfile not exists:{texts_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {texts_pkl}")
        with open(texts_pkl, 'rb') as inp:
            texts = pickle.load(inp)

    if not os.path.exists(txts_pkl):
        print(f"\t\tfile not exists:{txts_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {txts_pkl}")
        with open(txts_pkl, 'rb') as inp:
            txts = pickle.load(inp)

    # table элементы
    if not os.path.exists(tables_pkl):
        print(f"\t\tfile not exists:{tables_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {tables_pkl}")
        with open(tables_pkl, 'rb') as inp:
            tables = pickle.load(inp)

    if not os.path.exists(tbls_pkl):
        print(f"\t\tfile not exists:{tbls_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {tbls_pkl}")
        with open(tbls_pkl, 'rb') as inp:
            tbls = pickle.load(inp)

    # image элементы
    if not os.path.exists(images_pkl):
        print(f"\t\tfile not exists:{images_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {images_pkl}")
        with open(images_pkl, 'rb') as inp:
            images = pickle.load(inp)

    # summary text элементы
    if not os.path.exists(text_summaries_pkl):
        print(f"\t\tfile not exists:{text_summaries_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {text_summaries_pkl}")
        with open(text_summaries_pkl, 'rb') as inp:
            text_summaries = pickle.load(inp)

    # summary table элементы
    if not os.path.exists(table_summaries_pkl):
        print(f"\t\tfile not exists:{table_summaries_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {table_summaries_pkl}")
        with open(table_summaries_pkl, 'rb') as inp:
            table_summaries = pickle.load(inp)

    # img_base64_list элементы
    if not os.path.exists(img_base64_list_pkl):
        print(f"\t\tfile not exists:{img_base64_list_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {img_base64_list_pkl}")
        with open(img_base64_list_pkl, 'rb') as inp:
            img_base64_list = pickle.load(inp)

    # summary image элементы + meta
    if not os.path.exists(image_summaries_pkl):
        print(f"\t\tfile not exists:{image_summaries_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {image_summaries_pkl}")
        with open(image_summaries_pkl, 'rb') as inp:
            image_summaries = pickle.load(inp)

    # summary image элементы w/o meta
    if not os.path.exists(imgs_pkl):
        print(f"\t\tfile not exists:{imgs_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {imgs_pkl}")
        with open(imgs_pkl, 'rb') as inp:
            imgs = pickle.load(inp)

    # desc image base64 элементы w/o meta
    if not os.path.exists(image_desc_base64_pkl):
        print(f"\t\tfile not exists:{image_desc_base64_pkl}")
        exit(2)
    else:
        print(f"\t\tfile - ok: {image_desc_base64_pkl}")
        with open(image_desc_base64_pkl, 'rb') as inp:
            image_desc_base64 = pickle.load(inp)


    # печать сэмплов данных
    n_samples = 10
    print(f"\tprinting data samples [:{n_samples}]")

    if len(texts) > 0:
            print(f"\t\tlen(texts)={len(texts)}")
            print("\t\t\t", end="")
            print(texts[:n_samples])

    if len(txts) > 0:
            print(f"\t\tlen(txts)={len(txts)}")
            print("\t\t\t", end="")
            print(txts[:n_samples])

    if len(text_summaries) > 0:
            print(f"\t\tlen(text_summaries)={len(text_summaries)}")
            print("\t\t\t", end="")
            print(text_summaries[:n_samples])

    if len(tables) > 0:
            print(f"\t\tlen(tables)={len(tables)}")
            print("\t\t\t", end="")
            print(tables[:n_samples])

    if len(tbls) > 0:
            print(f"\t\tlen(tbls)={len(tbls)}")
            print("\t\t\t", end="")
            print(tbls[:n_samples])

    if len(table_summaries) > 0:
            print(f"\t\tlen(table_summaries)={len(table_summaries)}")
            print("\t\t\t", end="")
            print(table_summaries[:n_samples])

    if len(images) > 0:
            print(f"\t\tlen(images)={len(images)}")
            print("\t\t\t", end="")
            print(images[:n_samples])

    if len(image_summaries) > 0:
            print(f"\t\tlen(image_summaries)={len(image_summaries)}")
            print("\t\t\t", end="")
            print(image_summaries[:n_samples])

    if len(imgs) > 0:
            print(f"\t\tlen(imgs)={len(imgs)}")
            print("\t\t\t", end="")
            print(imgs[:n_samples])

    if len(image_desc_base64) > 0:
            print(f"\t\tlen(image_desc_base64)={len(image_desc_base64)}")
            print("\t\t\t", end="")
            print(image_desc_base64[:1])

    if len(img_base64_list) > 0:
            print(f"\t\tlen(img_base64_list)={len(img_base64_list)}")
    datetime_finish = datetime.datetime.now()
    delta_sec = date_diff_in_seconds(datetime_finish, start_datetime)
    el_d, el_h, el_m, el_s = dhms_from_seconds(delta_sec)
    print(f"{datetime_finish.strftime('%Y.%m.%d %H:%M:%S')} ->: end PDF file preprocessing statistics output in "
          f"{el_d} days {el_h} hours {el_m} min {el_s} sec")

gl_datetime_finish = datetime.datetime.now()
delta_sec = date_diff_in_seconds(gl_datetime_finish, gl_start_datetime)
el_d, el_h, el_m, el_s = dhms_from_seconds(delta_sec)
print(f"{gl_datetime_finish.strftime('%Y.%m.%d %H:%M:%S')} ->: preprocessing PDF file: {report_path} completed in "
      f"{el_d} days {el_h} hours {el_m} min {el_s} sec")

