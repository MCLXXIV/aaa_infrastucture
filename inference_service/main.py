from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import time


class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]
    inference_time_sec: float

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading model and tokenizer...')

    try:
        model_name = 'sergeyzh/rubert-mini-frida'

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        ml_models['tokenizer'] = tokenizer
        ml_models['model'] = model
        print('Model and tokenizer loaded successfully.')
    
    except Exception as e:
        print(f'Error loading model: {e}')
    
    yield 

    print('Cleaning up resources...')
    ml_models.clear()
    print('Resources cleaned up.')

app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    if 'model' in ml_models:
        return {'status': 'health', 'model': 'loaded'}
    return {'status': 'unhealthy', 'model': 'not loaded'}

@app.post('/embed', response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    tokenizer = ml_models.get('tokenizer')
    model = ml_models.get('model')

    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    start_time = time.perf_counter()

    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    end_time = time.perf_counter()
    pure_inference_time = end_time - start_time

    embedding = outputs.last_hidden_state[0, 0, :].tolist()

    return EmbedResponse(embedding=embedding,
                         inference_time_sec=pure_inference_time
                         )