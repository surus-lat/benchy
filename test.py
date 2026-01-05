import requests
import dotenv
import os
dotenv.load_dotenv()
SURUS_API_KEY = os.getenv("SURUS_API_KEY")
API_URL = "https://api.surus.dev/functions/v1/extract"
headers = {"Authorization": f"Bearer {SURUS_API_KEY}", "Content-Type": "application/json"}

data = {
    "text": "John Doe works at Acme Corp and can be reached at john@example.com",
    "json_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "company": {"type": "string"},
            "email": {"type": "string"}
        }
    }
}

response = requests.post(API_URL, headers=headers, json=data)
print(response.json())