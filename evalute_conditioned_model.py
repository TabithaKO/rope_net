import torch
import numpy as np
import argparse
from train_model import RopeToJointRegressor, RopeToConditionedJointRegressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained rope-to-joint models")
    parser.add_argument("--input_path", required=True, help="Path to input rope positions (.npy file)")
    parser.add_argument("--model_path", required=True, help="Path to trained model weights (.pth file)")
    parser.add_argument("--output_path", required=True, help="Path to save joint predictions (.npy file)")
    parser.add_argument("--model_type", choices=["base", "conditioned"], default="conditioned", 
                        help="Model type: 'base' (rope→joint) or 'conditioned' (rope→grasp→joint)")
    parser.add_argument("--save_grasp", action="store_true", 
                        help="Save predicted grasp positions (only for conditioned model)")
    parser.add_argument("--grasp_output_path", 
                        help="Path to save grasp predictions (.npy file), required if --save_grasp is set")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.save_grasp and args.model_type != "conditioned":
        parser.error("--save_grasp can only be used with conditioned model")
    
    if args.save_grasp and not args.grasp_output_path:
        parser.error("--grasp_output_path is required when --save_grasp is set")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model based on type
    if args.model_type == "base":
        print("Loading base model (rope→joint)...")
        model = RopeToJointRegressor(feat_size=128)
    else:
        print("Loading conditioned model (rope→grasp→joint)...")
        model = RopeToConditionedJointRegressor(feat_size=128)
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval().to(device)
    
    # Load input data
    print(f"Loading input data from {args.input_path}")
    data = np.load(args.input_path)
    inputs = torch.tensor(data, dtype=torch.float32).to(device)
    
    # Print input shape info
    print(f"Input shape: {inputs.shape}")
    if len(inputs.shape) == 4:
        batch_size, views, segments, coords = inputs.shape
        print(f"Batch size: {batch_size}, Views: {views}, Segments: {segments}, Coordinates: {coords}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        if args.model_type == "base":
            # Base model returns only joint predictions
            joint_preds = model(inputs)  # (N, 6, 9)
            grasp_preds = None
        else:
            # Conditioned model returns grasp and joint predictions
            grasp_preds, joint_preds = model(inputs)  # (N, 3) and (N, 6, 9)
    
    # Save joint predictions
    print(f"Saving joint predictions to {args.output_path}")
    np.save(args.output_path, joint_preds.cpu().numpy())
    
    # Save grasp predictions if requested
    if args.save_grasp and grasp_preds is not None:
        print(f"Saving grasp predictions to {args.grasp_output_path}")
        np.save(args.grasp_output_path, grasp_preds.cpu().numpy())
    
    print("Inference completed successfully.")
    print(f"Joint predictions shape: {joint_preds.shape}")
    if grasp_preds is not None:
        print(f"Grasp predictions shape: {grasp_preds.shape}")
        
        
# python evaluate_conditioned_model.py --input_path test_rope_data.npy --model_path checkpoints/best_conditioned_model.pth --output_path predicted_joints.npy --model_type conditioned --save_grasp --grasp_output_path predicted_grasps.npy