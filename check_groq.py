import requests
import os
import json

def check_groq_key(api_key: str) -> bool:
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [{"role": "user", "content": "Hello, is my API key working?"}],
        "model": "llama3-8b-8192"  # A free, fast model for testing
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for bad status codes
        
        data = response.json()
        print("Success! Response:", json.dumps(data, indent=2))
        return True
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {str(e)}")
        return False

# Usage
api_key = "GROQ_API_KEY"
if check_groq_key(api_key):
    print("API key is valid and working!")
else:
    print("API key check failed. See error above.")
