import os
import torch

def save_checkpoint(model, optimizer, epoch, save_dir, filename="checkpoint.pth"):
    """
    Save the model's state_dict and the training epoch along with the optimizer state.

    Parameters:
    - model: The trained model.
    - optimizer: The optimizer used for training.
    - epoch: The current epoch of training.
    - save_dir: Directory where the checkpoint will be saved.
    - filename: Name of the checkpoint file (default is "checkpoint.pth").
    """
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the full path for the checkpoint file
    checkpoint_path = os.path.join(save_dir, filename)

    # Save the model's state_dict, optimizer state, and other necessary information
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")
