import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

# Base point cloud encoder
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

# Dataset classes
class RopeToJointDataset(Dataset):
    def __init__(self, rope_data, joint_data):
        self.rope_data = rope_data  # shape: (N, 2, segment_count, 3)
        self.joint_data = joint_data  # shape: (N, 6, 9)

    def __len__(self):
        return len(self.rope_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.rope_data[idx], dtype=torch.float32),
            torch.tensor(self.joint_data[idx], dtype=torch.float32)
        )

class RopeToJointAndGraspDataset(Dataset):
    def __init__(self, rope_data, joint_data, grasp_data):
        """
        Dataset with rope positions, joint positions, and grasp positions
        
        Args:
            rope_data: shape (N, 2, segment_count, 3)
            joint_data: shape (N, 6, 9)
            grasp_data: shape (N, 3)
        """
        self.rope_data = rope_data
        self.joint_data = joint_data
        self.grasp_data = grasp_data
        
        # Get the segment count from the data
        _, _, self.segment_count, _ = rope_data.shape
        print(f"Dataset initialized with {self.segment_count} rope segments")

    def __len__(self):
        return len(self.rope_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.rope_data[idx], dtype=torch.float32),
            torch.tensor(self.joint_data[idx], dtype=torch.float32),
            torch.tensor(self.grasp_data[idx], dtype=torch.float32)
        )

# Model classes
class RopeToJointRegressor(nn.Module):
    """Base model that predicts joint positions from rope configurations"""
    def __init__(self, feat_size=128):
        super().__init__()
        self.encoder = SimplePointNetEncoder(input_dim=3, feat_size=feat_size)
        
        self.regressor = nn.Sequential(
            nn.Linear(2 * feat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 54)  # 6 x 9 joint positions
        )
        
    def forward(self, rope_input):
        pc1, pc2 = rope_input[:, 0], rope_input[:, 1]  # (B, segment_count, 3)
        f1 = self.encoder(pc1)
        f2 = self.encoder(pc2)
        combined = torch.cat([f1, f2], dim=1)
        out = self.regressor(combined)
        return out.view(-1, 6, 9)

class RopeToConditionedJointRegressor(nn.Module):
    """Advanced model that predicts grasp position first, then uses it to predict joint positions"""
    def __init__(self, feat_size=128):
        super().__init__()
        self.encoder = SimplePointNetEncoder(input_dim=3, feat_size=feat_size)
        
        # First predict the grasp point
        self.grasp_regressor = nn.Sequential(
            nn.Linear(2 * feat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # x, y, z grasp position
        )
        
        # Then predict joint positions conditioned on grasp point
        self.joint_regressor = nn.Sequential(
            nn.Linear(2 * feat_size + 3, 512),  # +3 for grasp position
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 54)  # 6 x 9 joint positions
        )
        
    def forward(self, rope_input):
        pc1, pc2 = rope_input[:, 0], rope_input[:, 1]  # (B, segment_count, 3)
        f1 = self.encoder(pc1)
        f2 = self.encoder(pc2)
        combined = torch.cat([f1, f2], dim=1)
        
        # First predict grasp point
        grasp_pred = self.grasp_regressor(combined)
        
        # Now predict joint positions conditioned on grasp point
        conditioned_input = torch.cat([combined, grasp_pred], dim=1)
        joint_pred = self.joint_regressor(conditioned_input)
        joint_pred = joint_pred.view(-1, 6, 9)
        
        return grasp_pred, joint_pred

# Training function for the base model
def train_base_model(rope_positions, joint_positions, num_epochs=1000):
    model = RopeToJointRegressor(feat_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Setup logging
    train_losses = []
    val_losses = []
    log_dir = "runs/rope_joint_regression"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Setup model checkpointing
    best_val_loss = float("inf")
    best_model_path = "checkpoints/best_rope_to_joint_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    # Create dataset and dataloader
    dataset = RopeToJointDataset(rope_positions, joint_positions)
    num_train = int(0.8 * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
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

        # Validation phase
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

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f" ---> Saved new best model (val loss: {avg_val_loss:.6f})")

        # Log metrics
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Save final model and results
    torch.save(model.state_dict(), "checkpoints/rope_to_joint_model.pth")
    torch.save(model, "checkpoints/full_model.pth")
    print("Saved the model weights to checkpoints")

    # Plot and save loss curves
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

    return model

# Training function for the conditioned model
def train_conditioned_model(rope_positions, joint_positions, grasp_positions, num_epochs=1000):
    model = RopeToConditionedJointRegressor(feat_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Loss functions
    joint_loss_fn = nn.MSELoss()
    grasp_loss_fn = nn.MSELoss()

    # Setup logging
    train_joint_losses = []
    train_grasp_losses = []
    train_total_losses = []
    val_joint_losses = []
    val_grasp_losses = []
    val_total_losses = []
    log_dir = "runs/rope_conditioned_regression"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Setup model checkpointing
    best_val_loss = float("inf")
    best_model_path = "checkpoints/best_conditioned_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    # Create dataset and dataloader
    dataset = RopeToJointAndGraspDataset(rope_positions, joint_positions, grasp_positions)
    num_train = int(0.8 * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_joint_loss = 0.0
        running_train_grasp_loss = 0.0
        running_train_total_loss = 0.0

        for rope_batch, joint_batch, grasp_batch in train_loader:
            rope_batch = rope_batch.to(device)
            joint_batch = joint_batch.to(device)
            grasp_batch = grasp_batch.to(device)
            
            # Forward pass
            grasp_pred, joint_pred = model(rope_batch)
            
            # Calculate losses
            grasp_loss = grasp_loss_fn(grasp_pred, grasp_batch)
            joint_loss = joint_loss_fn(joint_pred, joint_batch)
            
            # Weighted total loss (emphasize grasp accuracy)
            total_loss = 2.0 * grasp_loss + joint_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            batch_size = rope_batch.size(0)
            running_train_grasp_loss += grasp_loss.item() * batch_size
            running_train_joint_loss += joint_loss.item() * batch_size
            running_train_total_loss += total_loss.item() * batch_size

        # Calculate average training losses
        dataset_size = len(train_loader.dataset)
        avg_train_grasp_loss = running_train_grasp_loss / dataset_size
        avg_train_joint_loss = running_train_joint_loss / dataset_size
        avg_train_total_loss = running_train_total_loss / dataset_size
        
        train_grasp_losses.append(avg_train_grasp_loss)
        train_joint_losses.append(avg_train_joint_loss)
        train_total_losses.append(avg_train_total_loss)

        # Validation phase
        model.eval()
        running_val_grasp_loss = 0.0
        running_val_joint_loss = 0.0
        running_val_total_loss = 0.0
        
        with torch.no_grad():
            for rope_batch, joint_batch, grasp_batch in val_loader:
                rope_batch = rope_batch.to(device)
                joint_batch = joint_batch.to(device)
                grasp_batch = grasp_batch.to(device)
                
                # Forward pass
                grasp_pred, joint_pred = model(rope_batch)
                
                # Calculate losses
                grasp_loss = grasp_loss_fn(grasp_pred, grasp_batch)
                joint_loss = joint_loss_fn(joint_pred, joint_batch)
                total_loss = 2.0 * grasp_loss + joint_loss
                
                # Track losses
                batch_size = rope_batch.size(0)
                running_val_grasp_loss += grasp_loss.item() * batch_size
                running_val_joint_loss += joint_loss.item() * batch_size
                running_val_total_loss += total_loss.item() * batch_size
        
        # Calculate average validation losses
        dataset_size = len(val_loader.dataset)
        avg_val_grasp_loss = running_val_grasp_loss / dataset_size
        avg_val_joint_loss = running_val_joint_loss / dataset_size
        avg_val_total_loss = running_val_total_loss / dataset_size
        
        val_grasp_losses.append(avg_val_grasp_loss)
        val_joint_losses.append(avg_val_joint_loss)
        val_total_losses.append(avg_val_total_loss)

        # Save best model
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            torch.save(model.state_dict(), best_model_path)
            print(f" ---> Saved new best model (val loss: {avg_val_total_loss:.6f})")

        # Log metrics
        writer.add_scalar("Loss/Train/Grasp", avg_train_grasp_loss, epoch)
        writer.add_scalar("Loss/Train/Joint", avg_train_joint_loss, epoch)
        writer.add_scalar("Loss/Train/Total", avg_train_total_loss, epoch)
        writer.add_scalar("Loss/Val/Grasp", avg_val_grasp_loss, epoch)
        writer.add_scalar("Loss/Val/Joint", avg_val_joint_loss, epoch)
        writer.add_scalar("Loss/Val/Total", avg_val_total_loss, epoch)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | " 
              f"Train: Grasp={avg_train_grasp_loss:.6f}, Joint={avg_train_joint_loss:.6f}, "
              f"Total={avg_train_total_loss:.6f} | "
              f"Val: Grasp={avg_val_grasp_loss:.6f}, Joint={avg_val_joint_loss:.6f}, "
              f"Total={avg_val_total_loss:.6f}")

    # Save final model and results
    torch.save(model.state_dict(), "checkpoints/final_conditioned_model.pth")
    torch.save(model, "checkpoints/full_conditioned_model.pth")
    
    # Plot and save loss curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_total_losses, label='Train Total Loss')
    plt.plot(val_total_losses, label='Val Total Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(train_grasp_losses, label='Train Grasp Loss')
    plt.plot(val_grasp_losses, label='Val Grasp Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Grasp Prediction Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(train_joint_losses, label='Train Joint Loss')
    plt.plot(val_joint_losses, label='Val Joint Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Joint Prediction Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("loss_curves_conditioned.png")
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    # Load data
    rope_positions = np.load("rope_positions_3000.npy")
    joint_positions = np.load("joint_positions_3000.npy")
    grasp_positions = np.load("grasp_point_3000.npy")
    
    # Choose which model to train
    use_conditioned_model = True 
    
    if use_conditioned_model:
        print("Training conditioned model (grasp position → joint positions)...")
        model = train_conditioned_model(rope_positions, joint_positions, grasp_positions, num_epochs=5000)
    else:
        print("Training base model (rope positions → joint positions only)...")
        model = train_base_model(rope_positions, joint_positions, num_epochs=5000)