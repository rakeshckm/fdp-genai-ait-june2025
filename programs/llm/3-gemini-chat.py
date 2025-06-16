#https://ai.google.dev/gemini-api/docs/quickstart
from google import genai

client = genai.Client(api_key="AIzaSyB9u67Dxd3sXU9IfQPC79M98XCHF83UAhg")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)