import requests
import json

url = "http://localhost:8000/api/v1/auth/register"
payload = {
    "username": "http_user",
    "email": "http_user@example.com",
    "password": "password123"
}
response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
