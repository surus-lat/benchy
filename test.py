import requests

from dotenv import load_dotenv
import os

load_dotenv()
SURUS_API_KEY = os.getenv("SURUS_API_KEY")
API_URL = "https://api.surus.dev/functions/v1/ocr"
headers = {"Authorization": f"Bearer {SURUS_API_KEY}", "Content-Type": "application/json"}

import base64
image = 'src/tasks/image_extraction/.data/jpgs/20101090516_0003A00000838.jpg'
with open(image, "rb") as image_file:
    image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

data = {
    "image": f"data:image/jpeg;base64,{image_base64}",
    "prompt_default": "Extract structured data",
    "json_schema": {
        "type": "object",
        "properties": {
            "cuit_emisor": {"type": "string"},
            "razon_social_emisor": {"type": "string"}
        }
    }
}

response = requests.post(API_URL, headers=headers, json=data)
print(response.json())