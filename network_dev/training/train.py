import pickle
import sys
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np 


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from network_dev.models.hex_neural_net import HexNeuralNet

def load_data(batch_size=32):
    data = pickle.load(open("training_data_heuristic.pkl", "rb"))
    states = []
    policies = []
    values = []

    for state, policy, value in data:
        states.append(state)
        policies.append(policy)
        values.append(value)

    states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32)
    values_tensor = torch.tensor(np.array(values), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(states_tensor, policies_tensor, values_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def train(model, epochs=10, batch_size=32, lr=1e-3, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataloader = load_data(batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for states, target_policies, target_values in dataloader:
            states = states.to(device)                 # (B, 3, 11, 11)
            target_policies = target_policies.to(device)  # (B, 121)
            target_values = target_values.to(device)      # (B,)

            optimizer.zero_grad()

            # Forward pass
            pred_policy, pred_value = model(states)
            # pred_policy: (B, 121)
            # pred_value: (B, 1)

            # ----- Value loss (MSE) -----
            value_loss = ((pred_value.squeeze(1) - target_values) ** 2).mean()

            # ----- Policy loss (cross-entropy with distribution) -----
            policy_loss = -(
                target_policies * torch.log(pred_policy + 1e-8)
            ).sum(dim=1).mean()

            # ----- Total loss -----
            loss = value_loss + policy_loss

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {total_loss:.4f} | "
            f"Policy: {total_policy_loss:.4f} | "
            f"Value: {total_value_loss:.4f}"
        )

def create_train_and_save():
    model = HexNeuralNet()
    train(model)

    save_dir = PROJECT_ROOT / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "hex_neural_net.pth"
    torch.save(model, str(save_path))

    print(f"Model saved at {save_path}")
    return model


create_train_and_save()