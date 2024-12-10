from transformers import BertTokenizer, BertModel
import torch

def get_bert_embedding_pytorch(text):
    """
    Generate a text embedding using BERT and PyTorch.

    Parameters:
        text (str): The input text to embed.

    Returns:
        torch.Tensor: The embedding vector for the text.
    """
    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Tokenize the input text and convert to PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    input_ids = inputs["input_ids"]  # Token IDs
    attention_mask = inputs["attention_mask"]  # Attention mask
    
    # Pass inputs through the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Extract the last hidden state (shape: [batch_size, sequence_length, hidden_size])
    last_hidden_state = outputs.last_hidden_state
    
    # Mean pooling: Compute the mean of the token embeddings for a single vector
    embedding = torch.mean(last_hidden_state, dim=1)
    
    return embedding.squeeze()

if __name__ == "__main__":
    # Example text prompt
    text_prompt = "Artificial intelligence is transforming the world."

    # Get the BERT embedding using PyTorch
    embedding = get_bert_embedding_pytorch(text_prompt)

    # Print the results
    print(f"Text: {text_prompt}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")
