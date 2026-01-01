import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from unet import UNet
from diffusion import forward_diffusion_sample, T, sample
import os

# create a directory to save model output images
os.makedirs("output_images", exist_ok=True)

device="cuda" if torch.cuda.is_available() else "cpu"

transform=transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    transforms.RandomHorizontalFlip(p=0.5)
])

inv_transform = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
])

dataset=torchvision.datasets.STL10(root="data",split="train",transform=transform,download=True)
loader=DataLoader(dataset,batch_size=64,shuffle=True)

model=UNet(in_channels=3).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

epochs=50
save_interval=5 # save an image every 5 epochs
for epoch in range(epochs):
    for i, (images,labels) in enumerate(loader):
        optimizer.zero_grad()

        batch_size=images.shape[0]
        t=torch.randint(0,T,(batch_size,)).to(device)
        images=images.to(device)
        labels=labels.to(device)
        images_noised,noise=forward_diffusion_sample(images,t,device)

        predicted_noise=model(images_noised,t,labels)
        loss=nn.functional.mse_loss(predicted_noise,noise)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(loader)} Loss: {loss.item():.4f}")

    # check if we need to save output images
    if (epoch + 1) % save_interval == 0 or epoch == 0:
        print("Saving output images...")

        sample_labels=torch.tensor([i for i in range(10)],device=device)
        num_samples=len(sample_labels)

        with torch.no_grad():
            generated_images = sample(model, sample_labels, image_size=96, num_channels=3, batch_size=num_samples)

            save_path=f"output_images/epoch_{epoch+1}.png"
            torchvision.utils.save_image(generated_images, save_path, nrow=5, normalize=True)
            print(f"Saved generated images to {save_path}")


# save model checkpoint
checkpoint=torch.save(model.state_dict(),"unet_diffusion_stl10.pth")
