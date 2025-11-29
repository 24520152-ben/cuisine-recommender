import uvicorn
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = './model.onnx'
INGREDIENTS_PATH = '../data/ingredients.json'

with open(INGREDIENTS_PATH, 'r', encoding='utf-8') as f:
    all_ingredients = json.load(f)
num_features = len(all_ingredients)

ingredients_map = {name: idx for idx, name in enumerate(all_ingredients)}

ort_session = ort.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

class IngredientInput(BaseModel):
    selected_ingredients: list[str]

app = FastAPI()

@app.post('/predict')
def predict(data: IngredientInput):
    try:
        input_vector = np.zeros(num_features, dtype=np.float32)

        for ingredient in data.selected_ingredients:
            idx = ingredients_map[ingredient]
            input_vector[idx] = 1.0

        input_vector = input_vector.reshape(1, -1)

        outputs = ort_session.run(None, {input_name: input_vector})

        predicted_label = outputs[0][0]

        confidence = np.max(outputs[1][0])

        return {
            'cuisine': str(predicted_label),
            'confidence': float(confidence),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)