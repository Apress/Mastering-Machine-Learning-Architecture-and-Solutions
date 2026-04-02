import torch
from transformers import BertTokenizer, BertForSequenceClassification


# Define a function to use BERT for text classification
def classify_text(text, model, tokenizer, max_length=128):
    # Tokenize input text
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'  # PyTorch tensors
    )

    # Get model predictions
    with torch.no_grad():  # No gradient computation for inference
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    # Get predicted class (0 or 1 for binary classification)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class


# Example usage
def main():
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Sample text
    sample_text = "Transformers are revolutionizing machine learning!"

    # Classify the text
    prediction = classify_text(sample_text, model, tokenizer)

    # Print results
    print(f"Input text: {sample_text}")
    print(f"Predicted class: {prediction} (0 = negative, 1 = positive)")


# Run the example
if __name__ == "__main__":
    main()
