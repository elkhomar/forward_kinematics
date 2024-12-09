from pathlib import Path
import fire
from forward_kinematics import forward_kinematics
from tqdm import tqdm
from typing import Optional
import torch
import json

def convert_tensor_dict_to_serializable(data_dict):
    """Convert a dictionary containing PyTorch tensors to JSON-serializable format."""
    serializable_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            serializable_dict[key] = convert_tensor_dict_to_serializable(value)
        elif isinstance(value, torch.Tensor):
            serializable_dict[key] = value.detach().cpu().numpy().tolist()
        else:
            serializable_dict[key] = value
    return serializable_dict

def process_folder(
    input_dir: str,
    model_path: str = "/home/omar/Downloads/HumanGen.pkl",
    output_dir: Optional[str] = None
) -> None:
    """
    Process all NPZ files in a folder using forward kinematics.
    
    Args:
        input_dir: Path to folder containing NPZ files
        model_path: Path to HumanGen model file
        output_dir: Optional custom output directory path
    """
    input_path = Path(input_dir)
    if output_dir is None:
        output_path = Path(__file__).parent / "output"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    # Get all NPZ files in input directory
    npz_files = list(input_path.glob("*.npz"))
    
    # Get already processed files
    processed_files = {f.stem.replace("_3d_kp", "") for f in output_path.glob("*_3d_kp.json")}
    
    # Filter out already processed files
    remaining_files = [f for f in npz_files if f.stem not in processed_files]
    
    print(f"Found {len(npz_files)} total files")
    print(f"Already processed: {len(processed_files)} files")
    print(f"Remaining: {len(remaining_files)} files")
    
    # Process remaining files with progress bar
    for npz_file in tqdm(remaining_files, desc="Processing NPZ files"):
        try:
            export_dict = forward_kinematics(model_path=model_path, npz_path=str(npz_file))
            serializable_dict = convert_tensor_dict_to_serializable(export_dict)
            with open(output_path / f"{npz_file.stem}_3d_kp.json", "w") as f:
                json.dump(serializable_dict, f)
        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
            continue

if __name__ == "__main__":
    fire.Fire(process_folder)