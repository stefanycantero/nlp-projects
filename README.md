# Natural Language Processing Projects

Este repositorio reúne dos proyectos de **Procesamiento de Lenguaje Natural (PLN)** enfocados en el análisis, clasificación y modelado semántico de texto en español.  
Ambos desarrollan pipelines de NLP end-to-end: desde la adquisición y limpieza de datos hasta el entrenamiento, evaluación e interpretación de modelos basados en machine learning y deep learning.

---

## Estructura del repositorio
nlp-projects/

├── Sentiment_Analysis/

│ ├── data/

│ ├── notebooks/

│ ├── models/

│ └── README.md

└── News_Classification_Summarization/

├── data/

├── notebooks/

├── models/

└── README.md

Cada carpeta incluye:
- `data/`: dataset original y preprocesado.  
- `notebooks/`: flujo completo de trabajo en Jupyter.  
- `models/`: modelos entrenados y métricas de evaluación.  
- `README.md`: descripción específica del proyecto.
  
---
## Proyecto 1: Análisis de sentimientos en texto en español  

**Librerías:** `BeautifulSoup`, `Playwright`, `Scrapy`, `Pandas`, `NLTK`, `Scikit-learn`, `Matplotlib`, `Seaborn`

### Descripción
Proyecto enfocado en la detección automática de sentimientos en textos en español. Se construyó un corpus de más de **7,000 registros** mediante web scraping de foros y datasets de TASS, seguido de un proceso completo de **preprocesamiento lingüístico** (limpieza, tokenización, lematización, eliminación de stopwords y construcción de n-gramas).

Se aplicaron técnicas de **vectorización** con `CountVectorizer` y `TF-IDF`, y se entrenaron modelos de **clasificación supervisada** (`Regresión Logística`, `Árboles de Decisión`, `KNN`) para identificar la polaridad del texto.

### Flujo de trabajo
1. Extracción de datos vía scraping (BeautifulSoup, Scrapy, Playwright).  
2. Limpieza, tokenización y normalización del texto.  
3. Representación vectorial (TF-IDF y CountVectorizer).  
4. Entrenamiento de modelos con Scikit-learn.  
5. Evaluación de métricas (precision, recall, F1-score).  
6. Visualización de resultados con Seaborn y Matplotlib.

### Resultados
- Modelo seleccionado: **Regresión Logística con TF-IDF**  
- F1-score promedio: **84%**  
- Mayor estabilidad frente a desbalance de clases y vocabulario diverso.

---

## Proyecto 2: Clasificación, resumen y análisis temático de noticias  

**Librerías:** `Transformers (DistilBERT, T5-small)`, `Gensim`, `spaCy`, `NLTK`, `Scikit-learn`, `PyLDAvis`, `Matplotlib`, `Pandas`

### Descripción
Desarrollo de un sistema integral de **Procesamiento de Lenguaje Natural (PLN)** orientado a la **clasificación, resumen automático y análisis temático** de noticias en español.

Incluye tres módulos principales:
1. **Clasificación de texto** con modelo `DistilBERT-multilingual` (Hugging Face Transformers).  
2. **Generación de resúmenes automáticos** con el modelo `T5-small`.  
3. **Modelado de temas** con `LDA (Latent Dirichlet Allocation)` implementado en Gensim.

El flujo combina técnicas de *deep learning* y *topic modeling*, junto con visualización interactiva de resultados mediante `PyLDAvis`.

### Flujo de trabajo
1. Preprocesamiento lingüístico con `spaCy` y `NLTK`.  
2. Fine-tuning y evaluación del modelo `DistilBERT`.  
3. Generación de resúmenes automáticos con `T5-small` y evaluación con métricas ROUGE.  
4. Modelado de temas con `LDA` y visualización con `PyLDAvis`.  
5. Integración de módulos en un entorno interactivo mediante widgets en Jupyter Notebook.

### Resultados
- **Clasificación (DistilBERT):** F1-score = 80.16%  
- **Resumen automático (T5-small):** coherencia semántica validada con ROUGE.  
- **Modelado de temas (LDA):** coherencia ≈ 46%, visualización clara de tópicos predominantes.

