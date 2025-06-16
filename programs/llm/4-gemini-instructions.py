from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyB9u67Dxd3sXU9IfQPC79M98XCHF83UAhg")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a cat. Your name is Neko."),
    contents="Hello there"
)

print(response.text)