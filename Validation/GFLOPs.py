import torch
import torch.nn as nn
from thop import profile, clever_format
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.DAF_IRNet import DAF_IRNet

# Module wrapper instead of function
class ModelWrapper(nn.Module):
    def __init__(self, model, task_name):
        super().__init__()
        self.model = model
        self.task_name = task_name
    
    def forward(self, x):
        return self.model(x, task_names=[self.task_name])['output']

print("=" * 70)
print("DAFormer GFLOPs Measurement - All Tasks")
print("=" * 70)

# Initialize model
model = DAF_IRNet(input_channels=3, output_channels=3)
model.eval()

# Input configuration
input_tensor = torch.randn(1, 3, 256, 256)
tasks = ['shadow', 'denoise', 'rain', 'lol']

# Count parameters once (same for all tasks)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
print(f"Model Size (FP32): {total_params * 4 / (1024**2):.2f} MB")
print(f"Input Size: {input_tensor.shape}")
print()

# Measure GFLOPs for each task
results = {}

for task_name in tasks:
    print(f"Measuring {task_name.upper()}...")
    
    # module wrapper
    wrapped_model = ModelWrapper(model, task_name)
    wrapped_model.eval()
    
    try:
        flops, params = profile(wrapped_model, inputs=(input_tensor,), verbose=False)
        
        results[task_name] = {
            'flops': flops,
            'gflops': flops / 1e9,
            'macs': flops / 2e9,
            'params': params
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        results[task_name] = None

# Print results table
print("\n" + "-" * 70)
print(f"{'Task':<12} {'GFLOPs':<12} {'MACs (G)':<12} {'Parameters'}")
print("-" * 70)

for task_name in tasks:
    if results[task_name]:
        r = results[task_name]
        print(f"{task_name:<12} {r['gflops']:>6.2f} G    {r['macs']:>6.2f} G    {r['params']/1e6:>6.2f} M")
    else:
        print(f"{task_name:<12} {'Error':<12}")

print("-" * 70)

# Average
if all(results.values()):
    avg_gflops = sum(r['gflops'] for r in results.values()) / len(results)
    avg_macs = sum(r['macs'] for r in results.values()) / len(results)
    print(f"{'AVERAGE':<12} {avg_gflops:>6.2f} G    {avg_macs:>6.2f} G    {total_params/1e6:>6.2f} M")
    print("-" * 70)

print("\n GFLOPs measurement complete!")