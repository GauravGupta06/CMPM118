import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_snn(model, dataloader, optimizer, num_epochs=5, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = model(data)
            loss = F.cross_entropy(spk_rec.mean(0), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")
