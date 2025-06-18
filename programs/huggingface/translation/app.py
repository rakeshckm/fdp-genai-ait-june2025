#import spaces
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "sarvamai/sarvam-translate"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

#@spaces.GPU(duration=120)
def generate_response(tgt_lang, user_prompt):
    messages = [
        {"role": "system", "content": f"Translate the following sentence into {tgt_lang}."},
        {"role": "user", "content": user_prompt},
    ]
    
    # Apply chat template to structure the conversation
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    
    # Tokenize and move input to model device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate the output
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.01,
        num_return_sequences=1
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# Create Gradio UI
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Radio(["Hindi", "Bengali", "Marathi", "Telugu", "Tamil", "Gujarati", "Urdu", "Kannada", "Odia", "Malayalam", "Punjabi", "Assamese", "Maithili", "Santali", "Kashmiri", "Nepali", "Sindhi", "Dogri", "Konkani", "Manipuri (Meitei)", "Bodo", "Sanskrit"], label="Target Language", value="Hindi"),
        gr.Textbox(label="Input Text", value="Be the change you wish to see in the world."),
    ],
    outputs=gr.Textbox(label="Translation"),
    title="SARVAM - TRANSLATE",
    description="Now supporting 22 Indian languages and structured long-form text"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()