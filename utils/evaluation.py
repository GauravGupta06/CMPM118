import torch

def evaluate_snn(model, dataloader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = model(data)
            outputs = spk_rec.mean(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")
