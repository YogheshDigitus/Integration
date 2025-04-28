from transformers import CLIPModel, CLIPProcessor

# Load the model from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Optionally, save the model to a local directory
model.save_pretrained("/root/Chatbot/Integration/Transformers/")
processor.save_pretrained("/root/Chatbot/Integration/Transformers/")
