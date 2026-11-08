from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
for m in client.models.list():
    print(m.name)