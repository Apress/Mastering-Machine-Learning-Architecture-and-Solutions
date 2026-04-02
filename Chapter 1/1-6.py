import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# Data augmentation for SSL
transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

# Contrastive projection head
class ProjectionHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# SimCLR Model (backbone + head)
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = ProjectionHead(512)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

# NT-Xent Loss (Contrastive)
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.mm(z, z.t())

    labels = torch.arange(batch_size).repeat(2)
    labels = labels.to(z.device)

    mask = torch.eye(2 * batch_size).bool().to(z.device)
    similarity = similarity[~mask].view(2 * batch_size, -1)

    return F.cross_entropy(similarity / temperature, labels)

# Training Loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimCLR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    for imgs, _ in loader:
        imgs1 = imgs.to(device)
        imgs2 = imgs.to(device)

        z1 = model(imgs1)
        z2 = model(imgs2)

        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
