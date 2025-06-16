from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion pipeline for CPU
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32  # Use float32 for CPU
)
pipe = pipe.to("cpu")  # Force CPU usage

# Generate an image from a prompt
prompt = "a futuristic cityscape at sunset"
image = pipe(prompt).images[0]

# Save or display the image
image.save("generated_image.png")
image.show()