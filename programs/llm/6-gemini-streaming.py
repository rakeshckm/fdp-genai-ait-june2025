from google import genai

client = genai.Client(api_key="AIzaSyB9u67Dxd3sXU9IfQPC79M98XCHF83UAhg")

response = client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"]
)
for chunk in response:
    print(chunk.text, end="")