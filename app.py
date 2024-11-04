import keras
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from prometheus_fastapi_instrumentator import Instrumentator 


# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



description = """
    Cette application utilise un modèle de deep learning pour classifier des images de fruits. 
    Vous pouvez télécharger une image de fruit, et l'application affichera la classe correspondante 
    ainsi que le score de confiance de la prédiction.
"""

app = FastAPI(
    title = "Image Classification API",
    description = description,
    version = "0.1", 
    contact = {
        "name": "Abraham KOLOBOE",
        "email": "abklb27@gmail.com"
        }
)
image_size = (100, 100)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Instrumentator().instrument(app).expose(app)

def load_model():
    global model
    try :
        model = keras.models.load_model('model/best_model_cnn.keras')
        logging.info('Model loaded successfully')
    except Exception as e:
        logging.error(f'Error loading model: {e}')


def load_image_and_predict(image_path):
    try:
        logging.info(f"Loading and preprocessing the image from {image_path}.")
        img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        logging.info("Making prediction.")
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]

        return predicted_class, confidence
    except Exception as e:
        logging.error(f"Error in load_best_model_and_predict: {e}")
        raise
    
    
@app.on_event("startup")
async def startup_event():
    load_model()

def get_class_names():
    with open('class_names.txt', 'r') as file:
        class_names = file.readlines()
    return [class_name.strip() for class_name in class_names]

class_names = get_class_names()
logging.info('Class names loaded successfully')




@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"temp/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # class_names = sorted(os.listdir("data"))
        logging.info(f"Received image: {file.filename}")

        # Load the best model and make a prediction
        predicted_class, confidence = load_image_and_predict(file_location)
        predicted_class = class_names[predicted_class]

        # Clean up the temporary file
        os.remove(file_location)

        return JSONResponse(content={"predicted_class": predicted_class, "confidence": round(float(confidence)*100, 2)})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
