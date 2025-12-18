import pickle
import sys
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np 
import argparse
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from network_dev.models.hex_neural_net import HexNeuralNet

def load_data(batch_size=32, file_name="training_data_heuristic.pkl"):
    print(f"Loading data from {file_name}...")
    data = pickle.load(open(file_name, "rb"))
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

def train(model, epochs=10, batch_size=32, lr=1e-3, device=None,
          file_name="training_data_heuristic.pkl"):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataloader = load_data(batch_size=batch_size, file_name=file_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for states, target_policies, target_values in dataloader:
            states = states.to(device)                    # (B, 3, 11, 11)
            target_policies = target_policies.to(device)  # (B, 121)
            target_values = target_values.to(device)      # (B,)

            optimizer.zero_grad()

            # Forward pass
            pred_policy_logits, pred_value = model(states)
            # pred_policy_logits: (B, 121)
            # pred_value: (B, 1)

            # ----- Value loss -----
            value_loss = F.mse_loss(
                pred_value.squeeze(1),
                target_values
            )

            # ----- Policy loss (KL divergence) -----
            # policy_loss = F.kl_div(
            #     F.log_softmax(pred_policy_logits, dim=1),
            #     target_policies,
            #     reduction="batchmean"
            # )
            # policy_loss = -torch.sum(
            #     target_policies * F.log_softmax(pred_policy_logits, dim=1),
            #     dim=1
            # ).mean()

            eps = 1e-8

            policy_loss = -torch.mean(
                torch.sum(
                    target_policies * torch.log(pred_policy_logits + eps),
                    dim=1
                )
            )

            # ----- Total loss -----
            loss = value_loss + policy_loss

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
    print(f"Training complete.")

def create_train_and_save(file_name="training_data_self_play.pkl", batch_size=128, epochs=10, lr=1e-3):
    # def load_model():
    #         model_name="hex_neural_net.pth"
    #         # Project root (two levels up from this script)
    #         project_root = Path(__file__).resolve().parents[2]
    #         models_dir = project_root / "saved_models"
    #         model_path = models_dir / model_name   
    #         # Load the full saved model
    #         model = torch.load(model_path, map_location="cpu", weights_only=False)
    #         print("Neural network model loaded successfully.")
    #         return model
    model = HexNeuralNet()
    train(model, file_name=file_name, batch_size=batch_size, epochs=epochs, lr=lr)

    save_dir = PROJECT_ROOT / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "hex_neural_net.pth"
    torch.save(model, str(save_path))

    print(f"Model saved at {save_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="training_data_self_play.pkl",
        help="training data file"
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=128,
        help="batch size"
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate"
    )
    args = parser.parse_args()

    create_train_and_save(
    file_name=args.file,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr
    )

