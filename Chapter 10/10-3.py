from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests

# Load pre-trained Vision Transformer model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load and preprocess the image
url = 'https://example.com/image.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")

# Make a prediction
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class index:", predicted_class_idx)
