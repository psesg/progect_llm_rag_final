# progect_llm_rag_final
Проект является финальной работой курса Корпоративного университета Сбера "Большие языковые модели: от архитектур до построения мультимодальных систем"
- задача - создать мультимодальную RAG-систему на данных отчета Сбера за 2023 год в PDF формате
- результаты опуликованы здесь [progect_llm_rag_final](https://pse-project-rag-pure.streamlit.app/) 
### Общие сведения

- задача реализована созданием двух скриптов на Python, первый - **_prepare.py_** предобрабатывет входной файл и создает PKL файлы, используемыми затем вторым скриптом - **_rag_deploy.py_** для развертывания RAG pipeline системы и визуализации работы в WEB интерфейсе
- для визуализации работы pipeline RAG системы используется библиотека [streamlit](https://streamlit.io/) 
- скрипт **_prepare.py_** протестирован и может выполняться на ОС Windows 10/11 и Ubuntu 22.04+
- скрипт **_rag_deploy.py_** следует выполнять только на ОС Linux из-за проблем в Windows с совместимостью библиотек **_chromadb_**

### Установка ПО
- для установки и запуска проекта, необходим [Python](https://www.python.org/) (использовалась v3.12.4)  и для удобства - среда разработки [Pycharm](https://www.jetbrains.com/pycharm/).
- для работы скрипта **_prepare.py_** требуется установка дополнительного ПО 
  - [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html), например, для [Linux: так](https://ubuntu-news.ru/howto/raspoznavanie-teksta-v-ubuntu-kak-ustanovit-tesseract-ocr) и для [Windows: так](https://github.com/UB-Mannheim/tesseract/wiki)
  - [Poppler](https://poppler.freedesktop.org/) для Linux: sudo apt install poppler-utils
  - установка для Linux выполняется командами:
    - ```sh
       $ sudo apt install tesseract-ocr-rus 
       $ apt install libtesseract-dev
       $ sudo apt install poppler-utils
       ``` 
  - установка для Windows осуществляется из бинарных файлов:
    - Tesseract -[бинарники можно взять здесь](https://github.com/UB-Mannheim/tesseract/wiki)
    - Poppler - [бинарники можно взять здесь](https://github.com/oschwartz10612/poppler-windows)

### Настройка 
- в проекте используется модель от OpenAI "gpt-4o" и для ее использования в переменных окружения ОС необходимо настроить переменную OPENAI_API_KEY
- для правильной работы скрипта **_prepare.py_**  для предобработки входного файла должны должен быть созданы три подкаталога
  - подкаталог **_source_pdf_report_** - в нем должен быть помещен файл отчет Сбера в формате PDF
  - подкаталог **_pickles_** - в него скриптом **_prepare.py_**  будут записываться PKL файлы для дальнейшего их использования вторым скриптом - **_rag_deploy.py_**
  - подкаталог **_extracted_images_** - в него скриптом **_prepare.py_**  будут из файла PDF извлекаться изображения в JPG формате 
- для возможности локального запуска скрипта - **_rag_deploy.py_** и визуализации работы RAG pipeline системы с помощью библиотеки [streamlit](https://streamlit.io/) в корневом каталоге проекта должен быть создан директорий **_.streamlit_** (именно с точкой в начале имени) с файлом  **_secrets.toml_** содержащий ключ OPENAI_API_KEY
    - ```sh
         OPENAI_API_KEY="sk-proj-*****************************"
         ``` 
- установка библиотек Python (очень желательно установку производить в предварительно созданное виртуальное окружение)
  - для работы скрипта **_prepare.py_** достаточно установить библиотеки перечисленные в файле **_prequirements.txt_**
  - ```sh
     $ pip install prequirements.txt
     ``` 
  - для работы скрипта **_rag_deploy.py_** нужно установить библиотеки перечисленные в файле **_requirements.txt_**
  - ```sh
     $ pip install requirements.txt
     ``` 
### Запуск
- запуск скрипта **_prepare.py_** осуществляется из среды виртуального окружения командой
  -    ```sh
         $ puthon prepare.py 
          ``` 
- запуск скрипта **_rag_deploy.py_** осуществляется из среды виртуального окружения командой
  -    ```sh
         $ streamlit run rag_deploy.py 
       ```        
- для размещения RAG pipeline системы в интернете из репозитория GitHub можно воспользоваться сервисом публикации [streamlit](https://share.streamlit.io/)
