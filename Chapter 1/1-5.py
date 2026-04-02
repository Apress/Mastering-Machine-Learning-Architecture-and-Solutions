import torch
import torch.nn.functional as F

# Define sample embedding vectors
embedding1 = torch.tensor([0.1, 0.2, 0.3])
embedding2 = torch.tensor([0.1, 0.2, 0.35])

# Compute cosine similarity
similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
print("Similarity Score:", similarity.item())
