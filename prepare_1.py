# -*- coding: utf-8 -*-

from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from PIL import Image as pil_image
import os
import pickle
import base64
import warnings

warnings.filterwarnings('ignore')

# пути к файлам
# debug code on alternativnye-istochniki-energii.pdf work real on Sber2023.pdf
report_path = "source_pdf_report/Sber2023.pdf"
image_block_output_dir = "./extracted_images"
raw_pdf_elements_pkl = "./pickles/raw_pdf_elements_pkl.pkl"
texts_pkl = "./pickles/texts_pkl.pkl"
tables_pkl = "./pickles/tables_pkl.pkl"
texts_4k_token_pkl = "./pickles/texts_4k_token_pkl.pkl"
text_summaries_pkl = "./pickles/text_summaries_pkl.pkl"
table_summaries_pkl = "./pickles/table_summaries_pkl.pkl"
img_base64_list_pkl = "./pickles/img_base64_list.pkl"
image_summaries_pkl = "./pickles/image_summaries_pkl.pkl"

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
        extract_images_in_pdf=True,                     # Указание на то, что из PDF нужно извлечь изображения
        infer_table_structure=True,                     # Автоматическое определение структуры таблиц в документе
        chunking_strategy="basic",                      # Стратегия разбиения текста на части
        multipage_sections=False,                       # False - разделять элементы на разных страницах на отдельные фрагменты
        max_characters=1000,                            # Максимальное количество символов в одном чанке текста
        new_after_n_chars=800,                          # Число символов, после которого начинается новый чанк текста
        combine_text_under_n_chars=200,                 # Минимальное количество символов, при котором чанки объединяются
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=image_output_dir,# куда будут сохраняться извлеченные изображения
        # ocr_languages=["rus", "eng"],                 # языки для OCR - устарело
        languages=["rus", "eng"]                        # языки для текста
    )

# Функция категоризации элементов
def categorize_elements(raw_pdf_elements):
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
    for element in raw_pdf_elements:
        # Проверка типа элемента. Если элемент является таблицей, добавляем его в список таблиц
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        # Если элемент является композитным текстовым элементом, добавляем его в список текстов
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables  # Возвращаем списки с текстами и таблицами

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

    prompt = ChatPromptTemplate(prompt_text)
    # Создаем модель для генерации суммаризаций. Устанавливаем температуру 0 для детерминированных ответов.
    model = ChatOpenAI(temperature=0, model="gpt-4o") # OpenAI API ключ в os.environ["OPENAI_API_KEY"]

    # Определяем цепочку обработки запросов: сначала шаблон запроса, затем модель, затем парсер выходных данных
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []  # Список для хранения суммаризаций текстов
    table_summaries = []  # Список для хранения суммаризаций таблиц

    # Если есть текстовые элементы и требуется их суммирование
    if texts and summarize_texts:
        # Выполняем параллельное суммирование текстов
        text_summaries = summarize_chain.batch(texts, {"max_concurrency":5 })
    elif texts:
        # Если суммирование не требуется, просто передаем исходные тексты
        text_summaries = texts

    # Если есть таблицы, выполняем их суммирование
    if tables:
        # Выполняем параллельное суммирование таблиц
        table_summaries = summarize_chain.batch(tables, {"max_concurrency":5 })

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
def image_summarize(img_base64, prompt):
    """
    Функция для получения суммаризации изображения с использованием GPT модели.

    Аргументы:
    img_base64: Строка, изображение закодированное в формате base64.
    prompt: Строка, запрос для модели GPT, содержащий инструкцию для суммаризации изображения.

    Возвращает:
    Суммаризация изображения, возвращенная моделью GPT.
    """
    # Создаем объект модели GPT с заданными параметрами
    chat = ChatOpenAI(model="gpt-4o", max_tokens=3000) # OpenAI API ключ в os.environ["OPENAI_API_KEY"]

    # Отправляем запрос к модели GPT
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},  # Запрос для модели
                    {
                        "type": "image_url",  # Тип содержимого - изображение
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},  # Изображение в формате base64
                    },
                ]
            )
        ]
    )
    # Возвращаем содержимое ответа от модели
    return msg.content

def generate_img_summaries(path):
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
    image_summaries = []  # Список для хранения суммаризаций изображений

    # Запрос для модели GPT
    prompt = """Ты — специалист по созданию коротких и содержательных описаний по изображениям.
        Выполняй основные требования к создаваемому описанию по изображению:
        - опиши основные компоненты, предметы, их взаимное расположение цвета, формы и фон;
        - если на изображении или картинке содержится человек или несколько людей, то опиши их количество, пол, возраст, одежду;
        - если на изображении содержится только фон или мало графического контента, то просто сделай вывод о малоинформативном контенте;
    """
    # Обрабатываем все файлы в указанной директории
    list_files = os.listdir(path)
    n_files = len(list_files)
    n_file = 1
    for img_file in sorted(list_files):
        if img_file.endswith(".jpg"):  # Проверяем, что файл имеет расширение .jpg
            img_path = os.path.join(path, img_file)  # Полный путь к изображению
            is_file_reduced = False
            # if os.path.getsize(img_path) > 350000: # will reduce size on 0.5 due ChatOpenAI error too long query
            #     im = pil_image.open(img_path)
            #     new_width = int((float(im.size[0]) * 0.5))
            #     new_height = int((float(im.size[1]) * 0.5))
            #     new_dimensions = (new_width, new_height)
            #     resized_im = im.resize(new_dimensions, pil_image.LANCZOS)
            #     im.close()
            #     resized_im.save(img_path, optimize=True, quality=85)
            #     is_file_reduced = True
            base64_image = encode_image(img_path)  # Кодируем изображение в base64
            img_base64_list.append(base64_image)  # Добавляем закодированное изображение в список
            print(f"\timage processing \t#{n_file} from {n_files}\t{img_file}\t{is_file_reduced}\t{os.path.getsize(img_path)}", flush=True)
            image_summaries.append(image_summarize(base64_image, prompt))  # Получаем суммаризацию изображения
            n_file = n_file + 1

    return img_base64_list, image_summaries  # Возвращаем результаты

########################################################################################################################
# точка входа - начало реальной обработки файла
########################################################################################################################
if not os.path.exists(report_path):
    print(f"file for processing not exists:{report_path}")
    exit(1)

print(f"извлечениe элементов из PDF-файла: {report_path}")
# Извлекаем элементы из PDF-файла с помощью функции extract_pdf_elements
raw_pdf_elements = extract_pdf_elements(report_path, image_block_output_dir)

# сохраняем результаты для дальнейшего использования
with open(raw_pdf_elements_pkl, 'wb') as outp:
    pickle.dump(raw_pdf_elements, outp, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
print(f"категоризация элементов извлеченных из PDF-файла")
# Категоризируем извлеченные элементы на текстовые и табличные с помощью функции categorize_elements
texts, tables = categorize_elements(raw_pdf_elements)

# сохраняем результаты для дальнейшего использования
with open(texts_pkl, 'wb') as outp:
    pickle.dump(texts, outp, pickle.HIGHEST_PROTOCOL)

with open(tables_pkl, 'wb') as outp:
    pickle.dump(tables, outp, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
print(f"разбиваем объединенный текст на чанки")
# Создаем объект CharacterTextSplitter для разбиения текста на части (чанки)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1200,    # Максимальный размер чанка в символах
    chunk_overlap=150   # Количество перекрывающихся символов между чанками
)

# Объединяем все текстовые элементы в одну строку
joined_texts = " ".join(texts)

# Разбиваем объединенный текст на чанки, используя созданный CharacterTextSplitter
texts_4k_token = text_splitter.split_text(joined_texts)

# сохраняем результаты для дальнейшего использования
with open(texts_4k_token_pkl, 'wb') as outp:
    pickle.dump(texts_4k_token, outp, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
print(f"суммаризация текстов и таблиц извлеченных из PDF-файла")
# Вызываем функцию для суммаризации текстов и таблиц, указывая, что нужно суммировать тексты
text_summaries, table_summaries = generate_text_summaries(texts_4k_token, tables, summarize_texts=True)

# сохраняем результаты для дальнейшего использования
with open(text_summaries_pkl, 'wb') as outp:
    pickle.dump(text_summaries, outp, pickle.HIGHEST_PROTOCOL)

with open(table_summaries_pkl, 'wb') as outp:
    pickle.dump(table_summaries, outp, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
print(f"суммаризация изображений извлеченных из PDF-файла")
# Вызываем функцию для генерации суммаризаций изображений
img_base64_list, image_summaries = generate_img_summaries(image_block_output_dir)

# сохраняем результаты для дальнейшего использования
with open(img_base64_list_pkl, 'wb') as outp:
    pickle.dump(img_base64_list, outp, pickle.HIGHEST_PROTOCOL)
with open(image_summaries_pkl, 'wb') as outp:
    pickle.dump(image_summaries, outp, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
print(f"вывод статистики предобработки PDF-файла:")
print(f"\tпроверка сохраненных  pkl файлов")

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

# table элементы
if not os.path.exists(tables_pkl):
    print(f"\t\tfile not exists:{tables_pkl}")
    exit(2)
else:
    print(f"\t\tfile - ok: {tables_pkl}")
    with open(tables_pkl, 'rb') as inp:
        tables = pickle.load(inp)

# chank text элементы
if not os.path.exists(texts_4k_token_pkl):
    print(f"\t\tfile not exists:{texts_4k_token_pkl}")
    exit(2)
else:
    print(f"\t\tfile - ok: {texts_4k_token_pkl}")
    with open(texts_4k_token_pkl, 'rb') as inp:
        texts_4k_token = pickle.load(inp)

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

# summary image элементы
if not os.path.exists(image_summaries_pkl):
    print(f"\t\tfile not exists:{image_summaries_pkl}")
    exit(2)
else:
    print(f"\t\tfile - ok: {image_summaries_pkl}")
    with open(image_summaries_pkl, 'rb') as inp:
        image_summaries = pickle.load(inp)

# печать сэмплов данных
n_saples=10
if len(texts) > 0:
        print(f"\tlen(texts)={len(texts)}")
        print("\t", end="")
        print(texts[:n_saples])
if len(tables) > 0:
        print(f"\tlen(tables)={len(tables)}")
        print("\t", end="")
        print(tables[:n_saples])
if len(texts_4k_token) > 0:
        print(f"\tlen(texts_4k_token)={len(texts_4k_token)}")
        print("\t", end="")
        print(texts_4k_token[:n_saples])
if len(text_summaries) > 0:
        print(f"\tlen(text_summaries)={len(text_summaries)}")
        print("\t", end="")
        print(text_summaries[:n_saples])
if len(table_summaries) > 0:
        print(f"\tlen(table_summaries)={len(table_summaries)}")
        print("\t", end="")
        print(table_summaries[:n_saples])
if len(img_base64_list) > 0:
        print(f"\tlen(img_base64_list)={len(img_base64_list)}")
if len(image_summaries) > 0:
        print(f"\tlen(image_summaries)={len(image_summaries)}")
        print("\t", end="")
        print(image_summaries[:n_saples])
print(f"предобработка PDF-файла завершена")