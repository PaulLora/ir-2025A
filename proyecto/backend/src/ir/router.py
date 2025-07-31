from multiprocessing import context
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from src.api import route_version
from pathlib import Path
from PIL import Image
from io import BytesIO
import torch
import clip
from pydantic import BaseModel
import pandas as pd
import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
import base64
from enum import Enum
from typing import Optional

router = APIRouter()

# Configurar CLIP (cargar una sola vez al iniciar)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path para las imágenes del corpus
images_dir = Path(__file__).parent.parent.parent.parent / "data" / "corpus" / "Images"

# Load FAISS index
features_dir = Path(__file__).parent.parent.parent.parent / "data" / "features"
index_path = features_dir / "image_index.faiss"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Enums para tipos de búsqueda
class SearchType(str, Enum):
    IMAGE = "image"
    TEXT = "text"

# Modelo para request unificado
class UnifiedSearchRequest(BaseModel):
    text: Optional[str] = None
    search_type: SearchType

# Routes
@router.get("/process-corpus")
@route_version(major=1)
async def process_corpus():   
    # Read captions file
    df = await read_captions_file()    
    
    # Premake lists of paths and captions
    image_paths = [os.path.join(images_dir, fname) for fname in df['image'].tolist()]
    captions = df['caption'].tolist()
    
    # Coding images and captions
    image_features = []
    text_features = []
    for image_path, caption in zip(image_paths, captions):
        try:
            # Load and preprocess image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            vectorImage = await process_with_clip(image_bytes=image_bytes, type=SearchType.IMAGE)
            if not vectorImage['success']:
                print(f"Error processing image {image_path}: {vectorImage['error']}")
                continue
            else:
                image_features.append(vectorImage['vector'])
            
            # Tokenize and encode caption
            vectorCaption = await process_with_clip(text=caption, type=SearchType.TEXT)
            if not vectorCaption['success']:
                print(f"Error processing caption for {image_path}: {vectorCaption['error']}")
                continue
            else:
                text_features.append(vectorCaption['vector'])
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
        
    # Convert to numpy arrays
    image_features = np.array(image_features)
    text_features = np.array(text_features)

    # Save features to disk
    features_dir = Path(__file__).parent.parent.parent.parent / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    np.save(features_dir / "image_features.npy", image_features)
    np.save(features_dir / "text_features.npy", text_features)
    
    # Create FAISS index for image features
    index = faiss.IndexFlatL2(image_features.shape[1])  # L2 distance
    index.add(image_features.astype(np.float32))  # Convert to float32 for FAISS
    faiss.write_index(index, str(features_dir / "image_index.faiss"))
    
    return {"message": "Corpus processed successfully"}

@router.post("/seeker")
@route_version(major=1)
async def seeker(
    search_type: str = Form(...),
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    if search_type == SearchType.IMAGE:
        if not image.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo")
        content = await image.read()
    elif search_type == SearchType.TEXT:
        content = text.strip()
    try:
        # Load FAISS index        
        index = faiss.read_index(str(index_path))
        
        # Process with CLIP
        if search_type == SearchType.TEXT:
            query = await process_with_clip(text=content, type=SearchType.TEXT)
        elif search_type == SearchType.IMAGE:
            query = await process_with_clip(image_bytes=content, type=SearchType.IMAGE)

        k = 5  # Número de resultados a retornar
        query_vector = np.array(query['vector'], dtype=np.float32).reshape(1, -1)  # Reshape para FAISS
        distances, indices = index.search(query_vector, k)  # Búsqueda en el índice FAISS
        image_captions = await read_captions_file()  # Obtener resultados del DataFrame
        captions = image_captions['caption'].tolist()
        top_captions = [captions[i] for i in indices[0]]
        images_base64 = await images_to_base64(image_captions.iloc[indices[0]])
        prompt = f"Basado en las siguientes descripciones:\n" + "\n".join(top_captions) + "\n\nGenera una explicación o narrativa que integre esta información."

        response = client.responses.create(
            model="gpt-4.1",
            input=prompt
        )

        return {
            "similar_images": images_base64,
            "generative_response": response.output_text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

# Methods
async def process_with_clip(image_bytes: Optional[bytes] = None, text: Optional[str] = None, type: SearchType = SearchType.IMAGE):
    """
    Procesa la imagen con CLIP directamente desde memoria
    """
    try:
        if type == SearchType.IMAGE:
            image = Image.open(BytesIO(image_bytes)).convert("RGB").copy()
            content = preprocess(image).unsqueeze(0).to(device)
        elif type == SearchType.TEXT:
            content = clip.tokenize([text.strip()]).to(device)

        # Generar embeddings con CLIP
        with torch.no_grad():
            if type == SearchType.TEXT:
                features = model.encode_text(content)
            else:
                features = model.encode_image(content)
            # Normalizar para similitud coseno
            features = features / features.norm(dim=-1, keepdim=True)

        # Convertir a numpy para serialización
        vector = features.cpu().numpy().flatten().tolist()
        
        return {
            "success": True,
            "vector": vector,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error procesando con CLIP: {str(e)}"
        }
        
async def read_captions_file():
    """
    Leer el archivo de captions y devolver un DataFrame
    """    
    unprocessed_captions_file = Path(__file__).parent.parent.parent.parent / "data" / "corpus" / "unprocessed_captions.txt"
    captions_file = Path(__file__).parent.parent.parent.parent / "data" / "corpus" / "captions.txt"
    
    if captions_file.exists():
        df = pd.read_pickle(captions_file)
        return df
    
    with open(unprocessed_captions_file, 'r') as f:
        lines = f.readlines()

    # Process lines and create a DataFrame
    data = []    
    for line in lines[1:]:
        line = line.strip()
        if not line or ',' not in line:
            continue
        image, caption = line.split(',', 1)
        image = image.strip()
        caption = caption.strip()
        data.append({"image": image, "caption": caption})

    df = pd.DataFrame(data)
    
    # Save the DataFrame to a pickle file for future use
    df.to_pickle(captions_file)
    return df

async def images_to_base64(related_df):
    """ 
    Convertir imágenes a base64 desde un DataFrame
    """
    resultados = []
    for _, row in related_df.iterrows():
        image_path = images_dir / row['image']
        try:
            with open(image_path, "rb") as img_file:
                base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            resultados.append({
                "description": row.get("caption", ""),
                "imageb64": base64_str
            })
        except Exception as e:
            print(f"No se pudo procesar la imagen {image_path}: {e}")
    return resultados