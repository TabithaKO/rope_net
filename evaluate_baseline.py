import torch
import numpy as np
import argparse
from train_logreg import RopeToJointRegressor  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = RopeToJointRegressor(feat_size=128) 
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval().to(device)

    # Load input
    data = np.load(args.input_path)  # (N, 2, 53, 3)
    inputs = torch.tensor(data, dtype=torch.float32).to(device)

    # Run inference
    with torch.no_grad():
        preds = model(inputs)  # (N, 6, 9)

    # Save outputs
    np.save(args.output_path, preds.cpu().numpy())
    print(f"Saved predictions to {args.output_path}")
