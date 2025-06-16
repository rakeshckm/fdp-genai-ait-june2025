import cohere

# Replace with your Cohere API key
api_key = "xRUtfWFXb4fH4tyfWGVNaCpjZJUGHJT6EoUfWoDM"
co = cohere.Client(api_key)

print("Cohere Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = co.chat(message=user_input)
    print("Bot:", response.text)