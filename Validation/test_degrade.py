""" # TESTING SCRIPT WITH DEGRADED AREA DETECTION
"""

# For all tasks: python test.py --load 690
# For shadow task: python test.py --load 690 --tasks shadow
# For denoise task: test.py --load 690 --tasks denoise
# For rain task: python test.py --load 690 --tasks rain
# For LLIE task: python test.py --load 690 --tasks lol

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
import argparse
import time
import os
from tqdm import tqdm
from utils.data_loader import create_multitask_dataloaders
from models.DAF_IRNet import DAF_IRNet

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parser():
    parser = argparse.ArgumentParser(description='DAFormer Multi-Task Testing')
    parser.add_argument('-l', '--load', type=str, default='690', 
                       help='Checkpoint name (without SDAF_IRNet_ prefix and .pth extension)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('-o', '--out_path', type=str, default='./test_results_multitask')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('--tasks', nargs='+', default=['shadow', 'denoise', 'lol', 'rain'],
                       help='Tasks to test (shadow, denoise, lol, rain)')
    parser.add_argument('--individual_checkpoints', action='store_true',
                       help='Use individual task checkpoints instead of combined')
    return parser


def fix_model_state_dict(state_dict):
    """Fix model state dict from DataParallel"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


def unnormalize(x):
    x = x.transpose(1, 3)
    #mean, std
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x


def create_output_dirs(out_path, tasks):
    """Create output directories"""
    os.makedirs(out_path, exist_ok=True)
    
    # Combined results directory
    os.makedirs(os.path.join(out_path, 'combined'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'combined', 'comparison_grids'), exist_ok=True)
    
    # Individual task directories
    for task in tasks:
        task_dir = os.path.join(out_path, task)
        os.makedirs(task_dir, exist_ok=True)
        
        if task == 'denoise':
            for noise_level in [15, 25, 50]:
                noise_dir = os.path.join(task_dir, f'noise{noise_level}')
                os.makedirs(os.path.join(noise_dir, 'output_images'), exist_ok=True)
                os.makedirs(os.path.join(noise_dir, 'comparison_grids'), exist_ok=True)
                os.makedirs(os.path.join(noise_dir, 'degraded_areas'), exist_ok=True)  
        else:
            os.makedirs(os.path.join(task_dir, 'output_images'), exist_ok=True)
            os.makedirs(os.path.join(task_dir, 'comparison_grids'), exist_ok=True)
            os.makedirs(os.path.join(task_dir, 'degraded_areas'), exist_ok=True)  


def save_degradation_mask(shadow_masks, pred, save_path):
    """
    Save degradation detection mask visualization
    Args:
        shadow_masks: List of attention masks from model
        pred: Model output (for size reference)
        save_path: Path to save the mask image
    """
    import torch.nn.functional as F
    
    if len(shadow_masks) > 0:
        target_size = pred.shape[2:]
        
        # Resize all masks to same size and average
        resized_masks = []
        for mask in shadow_masks:
            if mask.shape[-2:] != target_size:
                mask_resized = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
            else:
                mask_resized = mask
            resized_masks.append(mask_resized)
        
        # Average all masks
        aggregated_mask = torch.stack(resized_masks).mean(dim=0)
        
        # Convert to 0-255 grayscale
        mask_np = aggregated_mask[0, 0].cpu().numpy()
        mask_np = (mask_np * 255).astype('uint8')
        
        # Save as grayscale image
        #from PIL import Image
        mask_img = Image.fromarray(mask_np, mode='L')
        mask_img.save(save_path)


def test_shadow(model, test_loader, device, out_path):
    """Test shadow removal"""
    print("\n" + "="*80)
    print("Testing Shadow Removal")
    print("="*80)
    
    model.eval()
    total_time = 0.0
    
    task_out = os.path.join(out_path, 'shadow')
    combined_out = os.path.join(out_path, 'combined')
    
    progress_bar = tqdm(test_loader, desc="Shadow")
    
    for batch_idx, (img, gt_shadow, gt, task_name) in enumerate(progress_bar):
        img_input = img.to(device)
        gt_input = gt.to(device)
        
        filename = test_loader.dataset.img_list['path_A'][batch_idx].split('/')[-1]
        base_name = os.path.splitext(filename)[0]
        
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # TASK-SPECIFIC: Pass task_names for MoE routing
                outputs = model(img_input, task_names=['shadow'])
                pred = outputs['output']
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # ============================================================
        # DEBUG BLOCK 1: Check if model is actually doing something
        # ============================================================
        if batch_idx < 3:  # Only print for first 3 images
            print(f"\n=== BATCH {batch_idx} DEBUG ===")
            print(f"Input range: [{img_input.min():.3f}, {img_input.max():.3f}]")
            print(f"Output range: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"Difference (abs): {(pred - img_input).abs().mean():.6f}")
            print(f"Input mean: {img_input.mean():.3f}, Output mean: {pred.mean():.3f}")
            
            # Check if output = input
            if (pred - img_input).abs().mean() < 0.001:
                print(" WARNING: Output ≈ Input! Model outputting identity!")
            else:
                print(" Model is changing the image")
        # ============================================================
        
        pred_cpu = pred.cpu()
        img_cpu = img_input.cpu()
        gt_cpu = gt_input.cpu()
        
        progress_bar.set_postfix({'Time': f'{inference_time*1000:.1f}ms'})
        
        # Save comparison grid (task-specific)
        grid = make_grid(
            torch.cat([unnormalize(img_cpu), unnormalize(gt_cpu), unnormalize(pred_cpu)], dim=0),
            nrow=3, padding=10, pad_value=1.0
        )
        save_image(grid, os.path.join(task_out, 'comparison_grids', f'{base_name}_comparison.jpg'))
        
        # Save comparison grid (combined)
        save_image(grid, os.path.join(combined_out, 'comparison_grids', f'shadow_{base_name}_comparison.jpg'))
        
        # Save output image
        pred_pil = transforms.ToPILImage()(unnormalize(pred_cpu)[0])
        pred_pil.save(os.path.join(task_out, 'output_images', filename))
        
        # ==================== Save degradation mask ====================
        shadow_masks = outputs.get('shadow_masks', [])
        if len(shadow_masks) > 0:
            mask_filename = f'{base_name}_degradation.png'
            save_degradation_mask(shadow_masks, pred, 
                                os.path.join(task_out, 'degraded_areas', mask_filename))
        # ==================== END  ====================
    
    avg_time = total_time / len(test_loader)
    print("\nStatistics:")
    print(f"  Images: {len(test_loader)}")
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  FPS: {1.0/avg_time:.2f}")


def test_denoise(model, test_loaders, device, out_path):
    """Test denoising"""
    print("\n" + "="*80)
    print("Testing Denoising")
    print("="*80)
    
    model.eval()
    
    task_out = os.path.join(out_path, 'denoise')
    combined_out = os.path.join(out_path, 'combined')
    
    for noise_level in [15, 25, 50]:
        print(f"\n  Noise Level: {noise_level}")
        test_loader = test_loaders[noise_level]
        
        total_time = 0.0
        noise_out = os.path.join(task_out, f'noise{noise_level}')
        
        progress_bar = tqdm(test_loader, desc=f"Noise {noise_level}")
        
        for batch_idx, (img, gt_shadow, gt, task_name) in enumerate(progress_bar):
            img_input = img.to(device)
            gt_input = gt.to(device)
            
            filename = os.path.basename(test_loader.dataset.img_list['path_A'][batch_idx])
            if not filename.endswith('.png'):
                filename = filename.replace('.jpg', '.png').replace('.JPG', '.png')
            base_name = os.path.splitext(filename)[0]
            
            start_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # TASK-SPECIFIC: Pass task_names for MoE routing
                    outputs = model(img_input, task_names=['denoise'])
                    pred = outputs['output']
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            pred_cpu = pred.cpu()
            img_cpu = img_input.cpu()
            gt_cpu = gt_input.cpu()
            
            progress_bar.set_postfix({'Time': f'{inference_time*1000:.1f}ms'})
            
            # Save comparison grid (task-specific)
            grid = make_grid(
                torch.cat([unnormalize(img_cpu), unnormalize(gt_cpu), unnormalize(pred_cpu)], dim=0),
                nrow=3, padding=10, pad_value=1.0
            )
            save_image(grid, os.path.join(noise_out, 'comparison_grids', f'{base_name}_comparison.jpg'))
            
            # Save comparison grid (combined)
            save_image(grid, os.path.join(combined_out, 'comparison_grids', f'denoise_n{noise_level}_{base_name}_comparison.jpg'))
            
            # Save output image
            pred_pil = transforms.ToPILImage()(unnormalize(pred_cpu)[0])
            pred_pil.save(os.path.join(noise_out, 'output_images', filename))
            
            # ====================  Save degradation mask ====================
            shadow_masks = outputs.get('shadow_masks', [])
            if len(shadow_masks) > 0:
                mask_filename = f'{base_name}_degradation.png'
                save_degradation_mask(shadow_masks, pred, 
                                    os.path.join(noise_out, 'degraded_areas', mask_filename))
            # ==================== END  ====================
        
        avg_time = total_time / len(test_loader)
        print(f"    Average time: {avg_time*1000:.2f}ms | FPS: {1.0/avg_time:.2f}")


def test_lol(model, test_loader, device, out_path):
    """Test low-light enhancement"""
    print("\n" + "="*80)
    print("Testing Low-Light Enhancement")
    print("="*80)
    
    model.eval()
    total_time = 0.0
    
    task_out = os.path.join(out_path, 'lol')
    combined_out = os.path.join(out_path, 'combined')
    
    progress_bar = tqdm(test_loader, desc="LOL")
    
    for batch_idx, (img, gt_shadow, gt, task_name) in enumerate(progress_bar):
        img_input = img.to(device)
        gt_input = gt.to(device)
        
        filename = test_loader.dataset.img_list['path_A'][batch_idx].split('/')[-1]
        base_name = os.path.splitext(filename)[0]
        
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # TASK-SPECIFIC: Pass task_names for MoE routing
                outputs = model(img_input, task_names=['lol'])
                pred = outputs['output']
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        pred_cpu = pred.cpu()
        img_cpu = img_input.cpu()
        gt_cpu = gt_input.cpu()
        
        progress_bar.set_postfix({'Time': f'{inference_time*1000:.1f}ms'})
        
        # Save comparison grid (task-specific)
        grid = make_grid(
            torch.cat([unnormalize(img_cpu), unnormalize(gt_cpu), unnormalize(pred_cpu)], dim=0),
            nrow=3, padding=10, pad_value=1.0
        )
        save_image(grid, os.path.join(task_out, 'comparison_grids', f'{base_name}_comparison.jpg'))
        
        # Save comparison grid (combined)
        save_image(grid, os.path.join(combined_out, 'comparison_grids', f'lol_{base_name}_comparison.jpg'))
        
        # Save output image
        pred_pil = transforms.ToPILImage()(unnormalize(pred_cpu)[0])
        pred_pil.save(os.path.join(task_out, 'output_images', filename))
        
        # ====================  Save degradation mask ====================
        shadow_masks = outputs.get('shadow_masks', [])
        if len(shadow_masks) > 0:
            mask_filename = f'{base_name}_degradation.png'
            save_degradation_mask(shadow_masks, pred, 
                                os.path.join(task_out, 'degraded_areas', mask_filename))
        # ==================== END  ====================
    
    avg_time = total_time / len(test_loader)
    print("\nStatistics:")
    print(f"  Images: {len(test_loader)}")
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  FPS: {1.0/avg_time:.2f}")


def test_rain(model, test_loader, device, out_path):
    """Test deraining"""
    print("\n" + "="*80)
    print("Testing Deraining")
    print("="*80)
    
    model.eval()
    total_time = 0.0
    
    task_out = os.path.join(out_path, 'rain')
    combined_out = os.path.join(out_path, 'combined')
    
    progress_bar = tqdm(test_loader, desc="Rain")
    
    for batch_idx, (img, gt_shadow, gt, task_name) in enumerate(progress_bar):
        img_input = img.to(device)
        gt_input = gt.to(device)
        
        filename = test_loader.dataset.img_list['path_A'][batch_idx].split('/')[-1]
        base_name = os.path.splitext(filename)[0]
        
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # TASK-SPECIFIC: Pass task_names for MoE routing
                outputs = model(img_input, task_names=['rain'])
                pred = outputs['output']
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        pred_cpu = pred.cpu()
        img_cpu = img_input.cpu()
        gt_cpu = gt_input.cpu()
        
        progress_bar.set_postfix({'Time': f'{inference_time*1000:.1f}ms'})
        
        # Save comparison grid (task-specific)
        grid = make_grid(
            torch.cat([unnormalize(img_cpu), unnormalize(gt_cpu), unnormalize(pred_cpu)], dim=0),
            nrow=3, padding=10, pad_value=1.0
        )
        save_image(grid, os.path.join(task_out, 'comparison_grids', f'{base_name}_comparison.jpg'))
        
        # Save comparison grid (combined)
        save_image(grid, os.path.join(combined_out, 'comparison_grids', f'rain_{base_name}_comparison.jpg'))
        
        # Save output image
        pred_pil = transforms.ToPILImage()(unnormalize(pred_cpu)[0])
        pred_pil.save(os.path.join(task_out, 'output_images', filename))
        
        # ==================== Save degradation mask ====================
        shadow_masks = outputs.get('shadow_masks', [])
        if len(shadow_masks) > 0:
            mask_filename = f'{base_name}_degradation.png'
            save_degradation_mask(shadow_masks, pred, 
                                os.path.join(task_out, 'degraded_areas', mask_filename))
        # ==================== END ====================
    
    avg_time = total_time / len(test_loader)
    print("\nStatistics:")
    print(f"  Images: {len(test_loader)}")
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  FPS: {1.0/avg_time:.2f}")


def load_model_checkpoint(checkpoint_path, model, device):
    """Load model checkpoint"""
    try:
        if os.path.isfile(checkpoint_path):
            # ============================================================
            # DEBUG BLOCK 3: Verify checkpoint before loading
            # ============================================================
            print(f"\n{'='*60}")
            print("DEBUG BLOCK 3: CHECKPOINT INFO")
            print(f"{'='*60}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
                if 'epoch' in checkpoint:
                    print(f"Trained to epoch: {checkpoint['epoch']}")
                if 'best_psnr' in checkpoint:
                    print(f"Best PSNR: {checkpoint['best_psnr']:.2f} dB")
                if 'optimizer_state_dict' in checkpoint:
                    print(" Contains optimizer state")
                
                # Check first weight value
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                first_key = list(state_dict.keys())[0]
                first_value = state_dict[first_key]
                print(f"First weight '{first_key}':")
                print(f"  Shape: {first_value.shape}")
                print(f"  Mean: {first_value.mean():.6f}")
                print(f"  Std: {first_value.std():.6f}")
                print(f"  Min/Max: [{first_value.min():.6f}, {first_value.max():.6f}]")
            else:
                print("Checkpoint is state_dict only (no metadata)")
                state_dict = checkpoint
                first_key = list(state_dict.keys())[0]
                first_value = state_dict[first_key]
                print(f"First weight '{first_key}': mean={first_value.mean():.6f}, std={first_value.std():.6f}")
            
            print(f"{'='*60}\n")
            # ============================================================
            
            # Try loading as full checkpoint first
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(fix_model_state_dict(checkpoint['model_state_dict']))
            else:
                model.load_state_dict(fix_model_state_dict(checkpoint))
            
            print(f" Loaded: {checkpoint_path}")
            
            # ============================================================
            # DEBUG BLOCK 2: Check model architecture after loading
            # ============================================================
            print(f"\n{'='*60}")
            print("DEBUG BLOCK 2: MODEL ARCHITECTURE CHECK")
            print(f"{'='*60}")
            
            # Check model architecture
            print(f"Model has GT pathway: {hasattr(model, 'gt_reconstruction_proj')}")
            print(f"Model has output_proj: {hasattr(model, 'output_proj')}")
            
            # Check if weights actually loaded
            first_param = next(model.parameters())
            print("\nFirst model parameter:")
            print(f"  Shape: {first_param.shape}")
            print(f"  Mean: {first_param.mean():.6f}")
            print(f"  Std: {first_param.std():.6f}")
            print(f"  Min/Max: [{first_param.min():.6f}, {first_param.max():.6f}]")
            
            # If all zeros, model didn't load correctly!
            if first_param.abs().max() < 0.0001:
                print(" WARNING: Model weights are near zero! Checkpoint didn't load properly!")
            else:
                print(" Model weights loaded successfully")
            
            # Count total parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("\nModel parameters:")
            print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
            
            print(f"{'='*60}\n")
            # ============================================================
            
            return True
        else:
            print(f" Not found: {checkpoint_path}")
            return False
    except Exception as e:
        print(f" Error loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(args):
    print("\n" + "="*80)
    print("DAFormer Multi-Task Testing")
    print("="*80 + "\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load test dataloaders
    print("Loading test datasets...")
    _, _, test_loaders, _, _ = create_multitask_dataloaders(
        batch_size=1,
        num_workers=4,
        size=args.image_size,
        crop_size=args.crop_size
    )
    
    # Filter tasks based on args
    available_tasks = list(test_loaders.keys())
    tasks_to_test = [task for task in args.tasks if task in available_tasks]
    
    if not tasks_to_test:
        print("No valid tasks to test!")
        return
    
    print(f"Tasks to test: {', '.join(tasks_to_test)}\n")
    
    # Create output directories
    create_output_dirs(args.out_path, tasks_to_test)
    
    # Model
    print("Initializing model...")
    model = DAF_IRNet(
        input_channels=3,
        output_channels=3,
        embed_dim=48,
        num_blocks=[2, 3, 3, 2],
        num_heads=[2, 4, 8, 8]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)\n")
    
    # Test each task
    for task in tasks_to_test:
        # Load checkpoint (combined or individual)
        if args.individual_checkpoints:
            checkpoint_name = f'{task}_{args.load}'
        else:
            checkpoint_name = args.load
        
        checkpoint_path = os.path.join(args.checkpoint_dir, f'DAF_IRNet_{checkpoint_name}.pth')
        
        print(f"\nLoading checkpoint for {task}: {checkpoint_path}")
        if not load_model_checkpoint(checkpoint_path, model, device):
            print(f"Skipping {task}...")
            continue
        
        # Test based on task
        if task == 'shadow':
            test_shadow(model, test_loaders['shadow'], device, args.out_path)
        elif task == 'denoise':
            test_denoise(model, test_loaders['denoise'], device, args.out_path)
        elif task == 'lol':
            test_lol(model, test_loaders['lol'], device, args.out_path)
        elif task == 'rain':
            test_rain(model, test_loaders['rain'], device, args.out_path)
    
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.out_path}/")
    print("\nStructure:")
    print("  - combined/comparison_grids/  : All tasks combined grids")
    for task in tasks_to_test:
        if task == 'denoise':
            print(f"  - {task}/noise15/  : Noise level 15 results")
            print(f"  - {task}/noise25/  : Noise level 25 results")
            print(f"  - {task}/noise50/  : Noise level 50 results")
        else:
            print(f"  - {task}/output_images/  : {task.capitalize()} output images")
            print(f"  - {task}/comparison_grids/  : {task.capitalize()} comparison grids")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)