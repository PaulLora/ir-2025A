from fastapi import APIRouter, File, UploadFile, HTTPException
from src.api import route_version
from pathlib import Path
import uuid
from PIL import Image
from io import BytesIO
import torch
import clip
from pydantic import BaseModel

router = APIRouter()

# Configurar CLIP (cargar una sola vez al iniciar)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Modelo para request body de texto
class TextSearchRequest(BaseModel):
    text: str

# Routes
@router.get("/test")
@route_version(major=1)
async def test():
    return {"status": "UP"}

@router.post("/search-by-image")
@route_version(major=1)
async def search_by_image(image: UploadFile = File(...), save_image: bool = False):
    # Configuración del directorio de uploads
    UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "data" / "upload"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Tipos de archivo permitidos
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Validar que se subió un archivo
    if not image.filename:
        raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo")
    
    # Validar extensión
    file_ext = Path(image.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de archivo no permitido. Extensiones válidas: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Validar tamaño
    content = await image.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Archivo demasiado grande. Máximo permitido: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Generar nombre único
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Guardar archivo
    try:
        # PROCESAR CON CLIP (directamente desde memoria)
        clip_results = await process_image_with_clip(content)
        
        response = {
            "message": "Imagen procesada exitosamente",
            "original_filename": image.filename,
            "size": len(content),
            "content_type": image.content_type,
            "clip_processing": clip_results
        }
        
        # GUARDAR SOLO SI SE SOLICITA
        if save_image:
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = UPLOAD_DIR / unique_filename
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            response.update({
                "saved": True,
                "filename": unique_filename,
                "path": str(file_path.relative_to(Path(__file__).parent.parent.parent.parent))
            })
        else:
            response["saved"] = False
        
        return response
    
    except Exception as e:
        # Si hay error, eliminar el archivo si se creó
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

@router.post("/search-by-text")
@route_version(major=1)
async def search_by_text(request: TextSearchRequest):
    """
    Convertir texto a vector CLIP para búsqueda semántica
    """
    text = request.text.strip()
    try:
        # Tokenizar texto
        text_input = clip.tokenize([text]).to(device)
        
        # Generar embeddings de texto
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        vector = text_features.cpu().numpy().flatten().tolist()
        
        return {
            "text": text,
            "vector_size": len(vector),
            "vector": vector,
            "model_used": "CLIP ViT-B/32"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando texto: {str(e)}")

# Methods
async def process_image_with_clip(image_bytes: bytes):
    """
    Procesa la imagen con CLIP directamente desde memoria
    """
    try:
        # Abrir imagen desde bytes (sin guardar)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Preprocesar para CLIP
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Generar embeddings con CLIP
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # Normalizar para similitud coseno
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convertir a numpy para serialización
        vector = image_features.cpu().numpy().flatten().tolist()
        
        return {
            "success": True,
            "vector_size": len(vector),
            "vector": vector,  # Vector de características de la imagen
            "model_used": "CLIP ViT-B/32",
            "device": device,
            "image_processed": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error procesando imagen con CLIP: {str(e)}"
        }