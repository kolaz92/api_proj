import PIL
from contextlib import asynccontextmanager
import random
import logging

logger = logging.getLogger('uvicorn.info')
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from utils.model_func import class_id_to_label, load_model, transform_image, text2toxicity
import torch

model, tmodel, tokenizer = None, None, None

# Create class of answer: class name and class index
class ImageClass(BaseModel):
    class_name: str
    class_index: int

class TextInput(BaseModel):
    text: str

class TextOutput(BaseModel):
    prob_of_tox: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model, tmodel, tokenizer
    model = load_model(True)
    logger.info('Model resnet loaded')
    tmodel, tokenizer = load_model(False)
    logger.info('Model tox loaded')
    yield
    # Clean up the ML models and release the resources
    del model, tmodel, tokenizer


app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    return 'Hello FastAPI!!'


@app.post('/classify')
def classify(file: UploadFile):
    image = PIL.Image.open(file.file) # open image
    adapted_image = transform_image(image) # preprocess image
    with torch.inference_mode():
        pred_index = model(adapted_image.unsqueeze(0)).numpy().argmax()
    imagenet_class = class_id_to_label(pred_index)
    # create response
    response = ImageClass(
        class_name=imagenet_class,
        class_index=pred_index
    )
    return response

@app.post('/clf_text')
def clf_text(data: TextInput):
    prob_of_tox = text2toxicity(data.text, tokenizer, tmodel)
    response = TextOutput(
        prob_of_tox=prob_of_tox
    )
    return response



##### 
# run from api folder:
# uvicorn app.main:app (--reload to check service after every saving)
# 
# check via cURL util:
# curl -X POST "http://127.0.0.1:8000/classify/" -L -H  "Content-Type: multipart/form-data" -F "file=@dog.jpeg;type=image/jpeg"
####