import torch
import time
import sys
import os

# Add the models directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.DAF_IRNet import DAF_IRNet

def measure_fps(model, input_tensor, task_name, device, num_warmup=20, num_iterations=100):
    """Measure FPS for a specific task"""
    
    # Warm-up
    with torch.no_grad():
        for _ in range(num_warmup):
            output_dict = model(input_tensor, task_names=[task_name])
            _ = output_dict['output']
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            output_dict = model(input_tensor, task_names=[task_name])
            _ = output_dict['output']
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.time()
            times.append((end - start) * 1000)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
    min_time = min(times)
    max_time = max(times)
    fps = 1000 / avg_time
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps
    }


# Configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
input_sizes = [(256, 256), (480, 640)]  # Different resolution tests
tasks = ['shadow', 'denoise', 'rain', 'lol']

print("=" * 70)
print("DAFormer FPS Measurement - All Tasks")
print("=" * 70)
print(f"Device: {device}")
print()

# Initialize model
model = DAF_IRNet(input_channels=3, output_channels=3).to(device)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
print()

# Test each input size
for input_size in input_sizes:
    h, w = input_size
    print("=" * 70)
    print(f"Testing with input size: {h}x{w}")
    print("=" * 70)
    
    # Create input tensor
    input_tensor = torch.randn(1, 3, h, w).to(device)
    
    # Test each task
    results = {}
    for task_name in tasks:
        print(f"\nMeasuring {task_name.upper()}...")
        results[task_name] = measure_fps(model, input_tensor, task_name, device)
    
    # Print results table
    print("\n" + "-" * 70)
    print(f"{'Task':<12} {'Avg Time (ms)':<18} {'FPS':<10} {'Min (ms)':<12} {'Max (ms)'}")
    print("-" * 70)
    
    for task_name in tasks:
        r = results[task_name]
        print(f"{task_name:<12} {r['avg_time']:>6.2f} ± {r['std_time']:<6.2f}    "
              f"{r['fps']:>6.2f}    {r['min_time']:>6.2f}      {r['max_time']:>6.2f}")
    
    print("-" * 70)
    
    # Average across all tasks
    avg_time_all = sum(r['avg_time'] for r in results.values()) / len(results)
    avg_fps_all = sum(r['fps'] for r in results.values()) / len(results)
    print(f"{'AVERAGE':<12} {avg_time_all:>6.2f}              {avg_fps_all:>6.2f}")
    print("-" * 70)
    print()

# Memory usage (if CUDA)
if torch.cuda.is_available():
    print("=" * 70)
    print("GPU Memory Usage")
    print("=" * 70)
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved: {reserved:.2f} MB")
    print("=" * 70)

print("\n All FPS measurements complete!")