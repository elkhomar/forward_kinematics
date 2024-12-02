import numpy as np
import os
from pathlib import Path
import fire


def print_npz_contents(npz_path: str) -> None:
    """
    Read and print the contents of an NPZ file.
    
    Args:
        npz_path: Path to the NPZ file
    """
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} does not exist.")
        return
        
    try:
        # Load the NPZ file
        data = np.load(npz_path)
        
        print(f"\nContents of {npz_path}:")
        print("-" * 50)
        
        # Print all arrays in the NPZ file
        for key in data.files:
            array = data[key]
            print(f"\nKey: {key}")
            print(f"Shape: {array.shape}")
            print(f"Type: {array.dtype}")
            #print(f"Data:\n{array}")
            
        data.close()
        
    except Exception as e:
        print(f"Error reading NPZ file: {e}")


if __name__ == "__main__":
    fire.Fire(print_npz_contents) 