import os
import sys
import torch
from ultralytics import YOLO

def train_yolov8():
    # Threading/BLAS safety for Windows
    os.environ['PYTORCH_ENABLE_MKL_FALLBACK'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'  # You can tune this if CPU overload

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n CUDA Available: {torch.cuda.is_available()}")
    if device == 'cuda':
        print(f" GPU: {torch.cuda.get_device_name(0)}")

    # Load YOLOv8 model
    model_path = 'yolov8m.pt'  # Use a smaller model for lower memory usage
    model = YOLO(model_path)

    # Training configuration
    print(f"\n Starting training on {model_path}")
    results = model.train(
        data='data.yaml',   # Path to your dataset config
        epochs=100,           # More training = better result
        imgsz=640,          # Further reduced image size for less memory usage
        batch=16,            # Further reduced batch size to fit GPU memory
        device=device,
        name='fire_detection',  # The name of the experiment
        save=True,
        save_period=10,      # Save every 10 epochs
        verbose=True,
        amp=True,            # Mixed precision training for faster training
        workers=4,           # Number of CPU threads to load data

        # Augmentations (disabled to save memory)
        mosaic=0.0,          # Disable mosaic augmentation
        hsv_h=0.0,           # Disable hue augmentation
        hsv_s=0.0,           # Disable saturation augmentation
        hsv_v=0.0,           # Disable value augmentation
        fliplr=0.5,          # Flip images horizontally (50% chance)
        flipud=0.0,          # Flip images vertically (0% chance)
        translate=0.1,       # Translate images (10% chance)
        scale=0.5,           # Scale images (50% chance)
        shear=0.0,           # Shear images (0% chance)
        perspective=0.0,     # Perspective distortion (0% chance)
        copy_paste=0.2,      # Copy-paste augmentation (20% chance)
        erasing=0.4,         # Random erasing (40% chance)

        # Optimization parameters
        optimizer='AdamW',   # AdamW optimizer
        lr0=0.001,           # Initial learning rate
        lrf=0.01,            # Learning rate final (after cosine decay)
        momentum=0.937,      # Momentum for SGD (ignored in AdamW)
        weight_decay=0.0005, # Weight decay for regularization
        dropout=0.0,         # Dropout rate (0.0 means no dropout)

        # Other parameters
        exist_ok=True,       # If experiment already exists, continue training
        plots=True,          # Show training plots
        val=True,            # Validate the model during training
    )

    print("\n Training completed!")
    print(f" Best model saved to: {results.save_dir}")

if __name__ == '__main__':
    # Ensure Windows compatibility with multiprocessing
    if sys.platform.startswith('win'):
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    
    # Set environment variable for handling CUDA memory issues
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    train_yolov8()
