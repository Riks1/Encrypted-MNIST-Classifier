import torch
import argparse
import pathlib
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
from model import MNISTNet

def build_loaders(batch_size: int, val_frac: float = 0.1):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    full_train = datasets.MNIST(root = './data',transform = transform,train = True,download = True)
    val_len = int(len(full_train) * val_frac)
    train_set, val_set = random_split(
        full_train,
        [len(full_train) - val_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
 
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False )
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
   model.eval()
   correct = total = 0
   for images, labels in loader:
      images, labels = images.to(device), labels.to(device)
      preds = model(images).argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += labels.size(0)
   model.train()
   return 100.0 * correct / total

def train(args: argparse.Namespace)->None:
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Device: {device}")
 
   train_loader, val_loader, test_loader = build_loaders(args.batch_size)
   model = MNISTNet().to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
   best_val_acc = 0.0
   save_path = pathlib.Path(args.save_path)
   save_path.parent.mkdir(parents=True, exist_ok=True)
   for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for images,labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()         
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        avg_loss = running_loss/len(train_loader.dataset)
        val_acc = evaluate(model,val_loader,device)
        scheduler.step(val_acc)
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            improved = "  ← saved"
   test_acc = evaluate(model, test_loader, device)
   print(f"\nTest accuracy (best checkpoint): {test_acc:.2f}%")
   print(f"Weights saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="encrypted")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save-path', type=str, default='mnist.pth', help='Checkpoint output path')
    train(parser.parse_args())