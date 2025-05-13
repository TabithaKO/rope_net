import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# from torch_points3d.models.classification.pointnet import PointNet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

class SimplePointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, feat_size=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, feat_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 3, N)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.max(x, 2)[0]  # (B, feat_size)

# class RopeToJointRegressor(nn.Module):
#     def __init__(self, feat_size=1024):
#         super().__init__()
#         self.encoder = PointNet(input_nc=3, feat_size=feat_size)
#         for param in self.encoder.parameters():
#             param.requires_grad = False
            
#         self.regressor = nn.Sequential(
#             nn.Linear(2 * feat_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 54)
#         )

#     def forward(self, rope_input):
#         pc1, pc2 = rope_input[:, 0], rope_input[:, 1]  # (B, 53, 3) each
#         f1 = self.encoder.forward_features(pc1)       # (B, 1024)
#         f2 = self.encoder.forward_features(pc2)
#         combined = torch.cat([f1, f2], dim=1)         # (B, 2048)
#         out = self.regressor(combined)
#         return out.view(-1, 6, 9)



class RopeToJointRegressor(nn.Module):
    def __init__(self, feat_size=128):
        super().__init__()
        self.encoder = SimplePointNetEncoder(input_dim=3, feat_size=feat_size)
        
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(2 * feat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 54)
        )

    def forward(self, rope_input):
        pc1, pc2 = rope_input[:, 0], rope_input[:, 1]  # (B, 53, 3)
        f1 = self.encoder(pc1)
        f2 = self.encoder(pc2)
        combined = torch.cat([f1, f2], dim=1)
        out = self.regressor(combined)
        return out.view(-1, 6, 9)


class RopeToJointDataset(Dataset):
    def __init__(self, rope_data, joint_data):
        self.rope_data = rope_data  # shape: (1000, 2, 53, 3)
        self.joint_data = joint_data  # shape: (1000, 6, 9)

    def __len__(self):
        return len(self.rope_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.rope_data[idx], dtype=torch.float32),
            torch.tensor(self.joint_data[idx], dtype=torch.float32)
        )

if __name__ == "__main__":
    model = RopeToJointRegressor(feat_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    num_epochs = 1000

    # Create TensorBoard log directory
    log_dir = "runs/rope_joint_regression"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float("inf")
    best_model_path = "checkpoints/best_rope_to_joint_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    rope_positions = np.load("rope_positions.npy")
    joint_positions = np.load("joint_positions.npy") 

    dataset = RopeToJointDataset(rope_positions, joint_positions)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_train = int(0.8 * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for rope_batch, joint_batch in train_loader:
            rope_batch, joint_batch = rope_batch.to(device), joint_batch.to(device)
            pred = model(rope_batch)
            loss = loss_fn(pred, joint_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * rope_batch.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for rope_batch, joint_batch in val_loader:
                rope_batch, joint_batch = rope_batch.to(device), joint_batch.to(device)
                pred = model(rope_batch)
                loss = loss_fn(pred, joint_batch)
                running_val_loss += loss.item() * rope_batch.size(0)

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f" ---> Saved new best model (val loss: {avg_val_loss:.6f})")

        # --- TensorBoard logging ---
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")


    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.abspath("loss_curve.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    torch.save(model.state_dict(), "checkpoints/rope_to_joint_model.pth")
    torch.save(model, "checkpoints/full_model.pth")
    print("Saved the model weights to checkpoints")

    # np.save("train_losses.npy", np.array(train_losses))
    # np.save("val_losses.npy", np.array(val_losses))



