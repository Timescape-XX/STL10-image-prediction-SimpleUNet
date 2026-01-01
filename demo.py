import torch
import torch.nn as nn
from unet import UNet
import matplotlib.pyplot as plt
from diffusion import sample,T
import os

"""
given a label of the cifar10 dataset, predict the image
example: input "dog", output the predicted image
"""

device="cuda" if torch.cuda.is_available() else "cpu"

pred_dict={
    "airplane":0,
    "bird":1,
    "car":2,
    "cat":3,
    "deer":4,
    "dog":5,
    "frog":6,
    "monkey":7,
    "ship":8,
    "truck":9
}

reverse_pred_dict={v:k for k,v in pred_dict.items()}

def predict_image(label,model_path="unet_diffusion_stl10.pth",image_size=96,num_channels=3):
    # convert label to index
    if label not in pred_dict:
        raise ValueError("Label not in STL-10 classes.")
    
    label_idx=pred_dict[label]

    # load pretrained model
    model = UNet(in_channels = 3).to(device)
    model.load_state_dict(torch.load(model_path,map_location = device))
    model.eval()

    # generate image
    with torch.no_grad():
        label_tensor=torch.full((1,),label_idx,device=device,dtype=torch.long)

        generated_image = sample(model, label_tensor, image_size=image_size, num_channels=num_channels,batch_size=1)

    generated_image = generated_image[0]
    
    return generated_image

os.makedirs("predicted_images", exist_ok=True)
def save_image(image_tensor, filename="predicted_image.png"):
    image_tensor = image_tensor.permute(1, 2, 0)

    if image_tensor.device.type == "cpu":
        image_np = image_tensor.numpy()
    else:
        image_np = image_tensor.cpu().numpy()
    
    plt.imsave(f"predicted_images/{filename}", image_np)
    print(f"Saved predicted image to predicted_images/{filename}")

def show_image(image_tensor):
    image_tensor = image_tensor.permute(1, 2, 0)

    if image_tensor.device.type == "cpu":
        image_np = image_tensor.numpy()
    else:
        image_np = image_tensor.cpu().numpy()
    
    plt.figure(figsize=(4,4))
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    print("Enter a label (type 'labels to see all labels): ")
    print("Enter 'exit' to quit.")
    input_label = input().strip()
    while input_label.lower() != "exit":
        if input_label.lower() == "labels":
            print("CIFAR-10 Labels:")
            for lbl in pred_dict.keys():
                print(f"- {lbl}")
        else:
            try:
                predicted_img = predict_image(input_label.lower())
                save_image(predicted_img, filename=f"{input_label}_predicted.png")
                show_image(predicted_img)
            except ValueError as e:
                print(e)
        
        print("Enter a label (type 'labels' to see all labels): ")
        print("Enter 'exit' to quit.")
        input_label = input().strip()