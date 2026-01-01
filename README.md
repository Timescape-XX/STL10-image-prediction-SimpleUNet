# A diffusion model implementation using UNet architecture
## This is a pretrained diffusion model.
We trained this model on the STL10 dataset, and it is able to predict images according to the label you given.
### Environment Setup
To give a try, please make sure you have installed pytorch. if not, please run:
```
pip3 install torch torchvision

# install remaining project requirements
pip install os matplotlib
```
We strongly recommend you to install on the GPU
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# install remaining project requirements
pip install os matplotlib
```
Or if you want to train or evaluate this model, make sure to install following python packages
```
pip install numpy matplotlib scikit-image os math
```
**Model checkpoints have already included in this repository**

### Run The Demo
Run the demo to see predicted images
```
python demo.py
```

### Output Files
You can check the terminal, after you typed in the label you want to predict, the output files will be stored in the predicted_images directory
Or if you are training the model, the output images during training are under output_images directory

### Notes
The result might not be quite well, if you want to improve the accuracy of output images, you can deepen the UNet architecture or run more epoches. 
We would appreciate any suggestions you make!
