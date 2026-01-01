import torch
import torch.nn as nn
from unet import UNet
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from diffusion import forward_diffusion_sample, T, sample, device
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

transform=transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

model_path = "unet_diffusion_stl10.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model=UNet(in_channels=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

dataset=torchvision.datasets.STL10(root="data",split="test",transform=transform,download=True)
dataloader=DataLoader(dataset,batch_size=64,shuffle=False)

inv_transform=transforms.Compose([
    transforms.Normalize((-1,-1,-1),(2,2,2)),
])

def evaluate_model(model, dataloader, num_generation_samples=100):
    """
    Args:
        model: prepared diffusion model
        dataloader: DataLoader for evaluation dataset
        num_generation_samples: number of samples to generate for quality metrics
    Returns:
        A dictionary containing average loss, psnr, and ssim metrics
    """
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_loss_samples = 0
    generation_samples_processed = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            batch_size = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            
            # calculate loss
            t = torch.randint(0, T, (batch_size,)).to(device)
            images_noised, noise = forward_diffusion_sample(images, t, device)
            predicted_noise = model(images_noised, t, labels)
            loss = nn.functional.mse_loss(predicted_noise, noise)
            total_loss += loss.item() * batch_size
            total_loss_samples += batch_size
            
            # evaluate generation quality
            if generation_samples_processed < num_generation_samples:
                # count batch size
                samples_to_process = min(batch_size, num_generation_samples - generation_samples_processed)
                gen_images = images[:samples_to_process]
                gen_labels = labels[:samples_to_process]
                
                # generate images using the diffusion model
                generated_images = sample(model, gen_labels, image_size=96, num_channels=3, batch_size=samples_to_process)
                
                # convert to numpy for metric calculation
                gen_images = inv_transform(gen_images).permute(0, 2, 3, 1).cpu().numpy()
                generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
                
                # ensure images are in [0, 1] range
                gen_images = np.clip(gen_images, 0, 1)
                generated_images = np.clip(generated_images, 0, 1)
                
                # calculate PSNR and SSIM
                for i in range(samples_to_process):
                    total_psnr += psnr(gen_images[i], generated_images[i], data_range=1.0)
                    total_ssim += ssim(gen_images[i], generated_images[i], data_range=1.0, channel_axis=2)
                
                generation_samples_processed += samples_to_process
    
    # calculate averages
    avg_loss = total_loss / total_loss_samples
    avg_psnr = total_psnr / generation_samples_processed if generation_samples_processed > 0 else 0
    avg_ssim = total_ssim / generation_samples_processed if generation_samples_processed > 0 else 0
    
    return {
        "loss": avg_loss,
        "psnr": avg_psnr,
        "ssim": avg_ssim
    }

if __name__ == "__main__":
    print("Preparing to evaluate the diffusion model...")
    print(f"Using device: {device}")

    metrics = evaluate_model(model, dataloader, num_generation_samples=100)
    
    print(f"\nEvaluation results:")
    print(f"Average evaluation loss: {metrics['loss']:.4f}")
    print(f"Average evaluation psnr: {metrics['psnr']:.4f} dB")
    print(f"Average evaluation ssim: {metrics['ssim']:.4f}")