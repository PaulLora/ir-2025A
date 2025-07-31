# Sistema de Recuperación de Información Visual con CLIP 🔍🖼️

Un sistema de búsqueda semántica multimodal que permite encontrar imágenes similares usando texto o imágenes como consulta. Implementado con CLIP de OpenAI, FAISS para búsqueda rápida y FastAPI como backend.

## 🚀 Características

- **🔍 Búsqueda por imagen**: Sube una imagen y encuentra imágenes similares en el corpus
- **📝 Búsqueda por texto**: Describe lo que buscas y encuentra imágenes relevantes
- **🧠 Búsqueda semántica**: Utiliza CLIP de OpenAI para embeddings multimodales
- **⚡ Búsqueda rápida**: Índice FAISS optimizado para consultas eficientes
- **🤖 Respuestas generativas**: Integración con OpenAI GPT para descripciones narrativas
- **🎨 Interfaz web**: Frontend interactivo con Vue.js y Vuetify
- **📊 Base64 images**: Imágenes codificadas para respuesta directa

## 🏗️ Arquitectura del Sistema

```
ir-2025A/
├── proyecto/
│   ├── backend/                 # API FastAPI
│   │   ├── src/
│   │   │   ├── ir/             # Módulo de recuperación de información
│   │   │   │   └── router.py   # Endpoints principales y lógica CLIP
│   │   │   ├── main.py         # Aplicación FastAPI principal
│   │   │   └── api.py          # Configuración de rutas y versioning
│   │   ├── pyproject.toml      # Configuración del proyecto Python
│   │   └── venv/              # Entorno virtual
│   ├── frontend/              # Interfaz web
│   │   ├── index.html         # SPA con Vue.js 3
│   │   ├── css/               # Estilos Vuetify
│   │   └── js/                # Bibliotecas JavaScript
│   └── data/                  # Datos del corpus y features
│       ├── corpus/
│       │   ├── Images/        # Dataset de imágenes
│       │   ├── captions.txt   # Descripciones originales
│       │   └── captions_processed.txt  # Cache pickle
│       └── features/          # Vectores y índices generados
│           ├── image_features.npy      # Embeddings de imágenes
│           ├── text_features.npy       # Embeddings de texto
│           └── image_index.faiss       # Índice de búsqueda
```

## 🛠️ Stack Tecnológico

### Backend
- **FastAPI** - Framework web moderno y rápido para Python
- **CLIP (ViT-B/32)** - Modelo multimodal de OpenAI para embeddings
- **FAISS** - Biblioteca de Facebook para búsqueda de similitud vectorial
- **PyTorch** - Framework de deep learning
- **OpenAI API** - Para generación de respuestas narrativas
- **Pandas** - Manipulación de datos del corpus
- **NumPy** - Operaciones vectoriales eficientes

### Frontend
- **Vue.js 3** - Framework JavaScript progresivo
- **Vuetify** - Biblioteca de componentes Material Design
- **Axios** - Cliente HTTP para comunicación con API

### Infraestructura
- **Poetry** - Gestión de dependencias Python
- **uvicorn** - Servidor ASGI de alto rendimiento

## 📋 Requisitos del Sistema

### Hardware Recomendado
- **GPU**: NVIDIA con CUDA 11.0+ (opcional pero recomendado)
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **Almacenamiento**: 5GB+ libres (modelos + datos + índices)
- **CPU**: Multi-core para procesamiento paralelo

### Software
- **Python**: 3.8 o superior
- **CUDA**: 11.0+ (si usas GPU)
- **Git**: Para clonar el repositorio

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/PaulLora/ir-2025A.git
cd ir-2025A/proyecto
```

### 2. Configurar Backend

#### Crear y activar entorno virtual
```bash
cd backend
python -m venv venv

# En Linux/Mac
source venv/bin/activate

# En Windows
venv\Scripts\activate
```

#### Instalar dependencias
```bash
# Instalar desde requirements.txt (ya generado)
pip install -r requirements.txt
```

#### Configurar variables de entorno
```bash
# Crear archivo .env en el directorio backend/
echo "OPENAI_API_KEY=tu_openai_api_key_aqui" > .env
```

### 3. Preparar Corpus de Datos

#### Estructura de datos esperada
```
data/corpus/
├── Images/              # Directorio con imágenes del dataset
│   ├── imagen1.jpg
│   ├── imagen2.png
│   ├── imagen3.gif
│   └── ...
└── captions.txt         # Archivo con descripciones
```

#### Formato del archivo captions.txt
```csv
image,caption
imagen1.jpg,"A brown dog running in the park"
imagen2.png,"Mountain landscape at sunset"
imagen3.gif,"A cat sleeping on a couch"
```

### 4. Ejecutar el Sistema

#### Iniciar el servidor backend
```bash
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

#### Servir frontend
El frontend está disponible directamente en:
```bash
# Acceder al frontend web
http://localhost/ir-2025A/proyecto/frontend/
```

> **Nota**: El frontend debe estar servido por un servidor web (Apache/Nginx) o accesible vía archivo local.

## 📖 Guía de Uso

> **⚠️ IMPORTANTE**: Antes de usar el sistema por primera vez, **DEBES** ejecutar el procesamiento del corpus. Sin este paso, las búsquedas fallarán.

### 1. Procesar el Corpus (Primera vez - OBLIGATORIO)
**¡PASO INICIAL REQUERIDO!** Antes de realizar cualquier búsqueda, procesa las imágenes y genera los índices:

```bash
curl -X GET "http://localhost:8000/v1/process-corpus"
```

Este proceso:
- Carga todas las imágenes del directorio
- Genera embeddings CLIP para imágenes y texto
- Crea índice FAISS optimizado
- Guarda cache para consultas rápidas

**Tiempo estimado**: 1-3 minutos por cada 1000 imágenes

### 2. Realizar Búsquedas

#### 🔍 Búsqueda por Texto
```bash
curl -X POST "http://localhost:8000/v1/seeker" \
  -F "search_type=text" \
  -F "text=a dog playing in the beach"
```

#### 🖼️ Búsqueda por Imagen
```bash
curl -X POST "http://localhost:8000/v1/seeker" \
  -F "search_type=image" \
  -F "image=@/path/to/your/image.jpg"
```

### 3. Interfaz Web
1. Abrir navegador en `http://localhost/ir-2025A/proyecto/frontend/`
2. Seleccionar tipo de búsqueda (texto o imagen)
3. Ingresar consulta o subir archivo
4. Ver resultados similares con descripción generativa de IA

## 🔧 Documentación de API

### Endpoints Principales

#### `GET /v1/process-corpus`
Procesa el corpus completo de imágenes y texto.

**URL Completa:** `http://localhost:8000/v1/process-corpus`

**Respuesta:**
```json
{
  "message": "Corpus processed successfully"
}
```

#### `POST /v1/seeker`
Endpoint unificado para búsqueda multimodal.

**URL Completa:** `http://localhost:8000/v1/seeker`

**Parámetros (Form Data):**
- `search_type`: "text" | "image"
- `text`: String de consulta (requerido si search_type=text)
- `image`: Archivo de imagen (requerido si search_type=image)

**Respuesta:**
```json
{
  "similar_images": [
    {
      "description": "A brown dog running in the park",
      "imageb64": "/9j/4AAQSkZJRgABAQAAAQ..."
    }
  ],
  "generative_response": "Las imágenes muestran escenas vibrantes de la naturaleza, donde los animales interactúan con su entorno de manera armoniosa..."
}
```

## ⚙️ Configuración Avanzada

### Parámetros del Modelo CLIP
```python
# En router.py
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU/CPU
model, preprocess = clip.load("ViT-B/32", device=device)  # Modelo CLIP
```

### Configuración de Búsqueda
```python
k = 5  # Número de resultados similares
# Tipos de índice FAISS disponibles:
# - IndexFlatL2: Exacto, lento para grandes datasets
```

### Variables de Entorno
```bash
# backend/.env
OPENAI_API_KEY=sk-...                    # API key de OpenAI (requerido)
```

## 🐛 Solución de Problemas Comunes

### ❌ Error: "Índice FAISS no encontrado"
```bash
# Solución: Procesar el corpus primero (PASO OBLIGATORIO)
curl -X GET "http://localhost:8000/v1/process-corpus"
```

### ❌ Error: CUDA out of memory
```python
# Solución: Cambiar a CPU en router.py
device = "cpu"
```

### ❌ Error: "No module named 'clip'"
```bash
# Solución: Instalar dependencias
pip install git+https://github.com/openai/CLIP.git
```

### ❌ Error: OpenAI API
```bash
# Solución: Verificar API key
echo $OPENAI_API_KEY
# Debe empezar con 'sk-'
```

### ❌ Imágenes no cargan
```bash
# Verificar estructura de archivos
ls data/corpus/Images/     # Debe contener imágenes
ls data/corpus/captions.txt  # Debe existir archivo captions
```

### ❌ Performance lenta
- Usar GPU si está disponible
- Reducir tamaño del corpus para pruebas
- Verificar que FAISS use índice optimizado
- Monitorear uso de memoria RAM

## 📊 Rendimiento y Benchmarks

### Tiempos de Respuesta (Hardware típico)

| Operación | CPU (i7) | GPU (RTX 3070) |
|-----------|----------|----------------|
| Carga inicial | 30s | 15s |
| Procesamiento (1K imágenes) | 5-8 min | 2-3 min |
| Búsqueda individual | 200-500ms | 100-300ms |
| Carga de índice FAISS | 1-3s | 1-2s |

### Uso de Memoria

| Componente | RAM Utilizada |
|------------|---------------|
| Modelo CLIP | ~1GB |
| Índice FAISS (10K imágenes) | ~200MB |
| Cache de captions | ~10-50MB |
| **Total estimado** | **~1.5GB** |

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. **Fork** el proyecto
2. **Crear rama**: `git checkout -b feature/nueva-funcionalidad`
3. **Commit**: `git commit -am 'Agregar nueva funcionalidad'`
4. **Push**: `git push origin feature/nueva-funcionalidad`
5. **Pull Request**: Describir cambios claramente

### Áreas de Mejora
- [ ] Soporte para más formatos de imagen
- [ ] Traducción automática de captions
- [ ] Búsqueda combinada (imagen + texto)
- [ ] Dashboard de métricas
- [ ] API rate limiting
- [ ] Dockerización completa
- [ ] Tests automatizados

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver [LICENSE](LICENSE) para más detalles.

## 👥 Equipo

- **Diego Suquillo** - [@dsuquillo](https://github.com/dsuquillo)
- **Byron Carpio** - [@bcarpio16](https://github.com/bcarpio16)
- **Paul Lora** - [@PaulLora](https://github.com/PaulLora)

## 🙏 Reconocimientos

- **OpenAI** - Por el increíble modelo CLIP
- **Facebook Research** - Por la biblioteca FAISS
- **FastAPI Team** - Por el excelente framework web
- **Vue.js Team** - Por el framework frontend reactivo
- **Vuetify Team** - Por los componentes Material Design

## 📚 Referencias Técnicas

- [CLIP: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)
- [FAISS: A Library for Efficient Similarity Search](https://faiss.ai/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Vue.js 3 Guide](https://vuejs.org/guide/)
- [Vuetify: Vue Component Framework](https://vuetifyjs.com/)

## 🚀 Inicio Rápido (TL;DR)

```bash
# 1. Clonar y configurar
git clone https://github.com/PaulLora/ir-2025A.git
cd ir-2025A/proyecto/backend
python -m venv venv && source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar OpenAI
echo "OPENAI_API_KEY=tu_key_aqui" > .env

# 4. Iniciar servidor
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 5. PRIMER PASO OBLIGATORIO: Procesar corpus
curl -X GET "http://localhost:8000/v1/process-corpus"

# 6. Abrir interfaz web
# Navegar a: http://localhost/ir-2025A/proyecto/frontend/
```

**🎉 ¡Sistema listo!** 

- **API Base**: `http://localhost:8000/v1/`
- **Frontend**: `http://localhost/ir-2025A/proyecto/frontend/`

---

> **Nota**: Este proyecto es parte del curso de Recuperación de Información 2025A. Desarrollado con fines educativos y de investigación.
