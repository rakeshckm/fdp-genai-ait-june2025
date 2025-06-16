from PIL import Image
from google import genai

client = genai.Client(api_key="AIzaSyB9u67Dxd3sXU9IfQPC79M98XCHF83UAhg")

image = Image.open("D:\\kaushalya\\consultancy\\aitchikkamagalore\\sandbox\\generated_image.png")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, "Tell me about this image"]
)
print(response.text)