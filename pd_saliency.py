import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
from skimage import io, transform
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms, utils, models
from sklearn.preprocessing import label_binarize

import cv2
import numpy as np

class Gradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))

            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class BackProp:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = Gradients(
            self.model, target_layers, reshape_transform)

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # Forward propagation obtains the network output logits (without going through softmax).
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        #In most of the saliency attribution papers, the saliency is computed with a single target layer.
        #Commonly it is the last convolutional layer. Here we support passing a list with multiple target layers.
        #It will compute the saliency image for every image, and then aggregate them (with a default mean aggregation).
        #This gives us more flexibility in case we just want to use all conv layers for example, all Batchnorm layers, or something else.

        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_grad(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def crop_imgs(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img

weights_path = "Downloads/PD_vgg16/pd_vgg16_model_fold_0.pth"

model_pre.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

model_pre

target_layers = [model_pre.features[-1]]
data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

# load image
img_path = "Downloads/sample.png"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path).convert('RGB')

img_tensor = data_transform(img)

# expand batch dimension
input_tensor = torch.unsqueeze(img_tensor, dim=0)

model_pre = models.vgg16()

# Load model weights
weights_path = "Downloads/PD_vgg16/pd_vgg16_model_fold_0.pth"
model_pre.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

# Load and preprocess the input image
img_path = "Downloads/74-8.jpg"

import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

# Define a function for center cropping (if necessary)
def crop_imgs(img, new_size):
    height, width = img.shape[:2]
    startx = width//2 - new_size//2
    starty = height//2 - new_size//2
    return img[starty:starty+new_size, startx:startx+new_size]

# Load model weights
weights_path = "Downloads/PD_vgg16/pd_vgg16_model_fold_0.pth"
model_pre.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

# Load and preprocess the input image
img_path = "Downloads/saliency_map/74-8.jpg"
assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
img = Image.open(img_path).convert('RGB')

# Assuming data_transform is a predefined transformation function
img_tensor = data_transform(img)
input_tensor = torch.unsqueeze(img_tensor, dim=0)

# Create BackProp object
cam = BackProp(model=model_pre, target_layers=target_layers, use_cuda=True)
target_category = 0

# Obtain the heatmap
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
grayscale_cam = grayscale_cam[0, :]

# Process original image for visualization
img_array = np.array(img)

# Resize the heatmap to match the original image size
resized_cam = cv2.resize(grayscale_cam, (img_array.shape[1], img_array.shape[0]))

# Visualize and save the result
visualization = show_grad(img_array.astype(dtype=np.float32) / 255., resized_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()

# Save the original image
original_save_path = "Downloads/sample_original_output.png"
plt.imshow(img)
plt.savefig(original_save_path, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()

# Save the visualization
save_path = "Downloads/sample_output.png"
# plt.imshow(visualization)
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=True)

grayscale_cam
plt.matshow(grayscale_cam)
plt.grid(True)
plt.show()

import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

# Define a function for center cropping (if necessary)
def crop_imgs(img, new_size):
    height, width = img.shape[:2]
    startx = width//2 - new_size//2
    starty = height//2 - new_size//2
    return img[starty:starty+new_size, startx:startx+new_size]

# Load model weights
weights_path = "Downloads/PD_vgg16/pd_vgg16_model_fold_0.pth"
model_pre.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

# Load and preprocess the input image
img_path = "Downloads/saliency_map/71-1.jpg"
assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
img = Image.open(img_path).convert('RGB')

# Assuming data_transform is a predefined transformation function
img_tensor = data_transform(img)
input_tensor = torch.unsqueeze(img_tensor, dim=0)

# Create BackProp object
cam = BackProp(model=model_pre, target_layers=target_layers, use_cuda=True)
target_category = 1

# Obtain the heatmap
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
grayscale_cam = grayscale_cam[0, :]

# Process original image for visualization
img_array = np.array(img)
# Uncomment the next line if center cropping is required
# img_array = crop_imgs(img_array, 224)

# Resize the heatmap to match the original image size
resized_cam = cv2.resize(grayscale_cam, (img_array.shape[1], img_array.shape[0]))

# Visualize and save the result
visualization = show_grad(img_array.astype(dtype=np.float32) / 255., resized_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()

# Save the original image
original_save_path = "Downloads/sample_original_output.png"
plt.imshow(img)
plt.savefig(original_save_path, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()

# Save the visualization
save_path = "Downloads/sample_output.png"
# plt.imshow(visualization)
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=True)

grayscale_cam
plt.matshow(grayscale_cam)
plt.grid(True)
plt.show()

weights_path = "Downloads/PD_vgg16/pd_vgg16_model.pth"
# weights_path = "Downloads/model_0.pth"
model_pre.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

# load image
img_path = "Downloads/7_1.jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path).convert('RGB')

img_tensor = data_transform(img)
input_tensor = torch.unsqueeze(img_tensor, dim=0)

cam = BackProp(model=model_pre, target_layers=target_layers, use_cuda=True)
target_category = 0
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

grayscale_cam = grayscale_cam[0, :]

img_array = np.array(img)
img_array = crop_imgs(img_array, 224)
visualization = show_grad(img_array.astype(dtype=np.float32) / 255.,grayscale_cam,use_rgb=True)
plt.imshow(visualization)
plt.show()

weights_path = "Downloads/PD_vgg16/pd_vgg16_model_fold_3.pth"

# Save the visualization
save_path = "Downloads/sample_output.png"
plt.imshow(visualization)
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()

cam = BackProp(model=model_pre, target_layers=target_layers, use_cuda=True)
target_category = 1
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

grayscale_cam = grayscale_cam[0, :]

img_array = np.array(img)
img_array = crop_imgs(img_array, 224)
visualization = show_grad(img_array.astype(dtype=np.float32) / 255.,grayscale_cam,use_rgb=True)
plt.imshow(visualization)
plt.show()

cam = BackProp(model=model_pre, target_layers=target_layers, use_cuda=True)
target_category = 1
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

grayscale_cam = grayscale_cam[0, :]

img_array = np.array(img)
img_array = crop_imgs(img_array, 224)
visualization = show_grad(img_array.astype(dtype=np.float32) / 255.,grayscale_cam,use_rgb=True)
plt.imshow(visualization)
plt.show()

grayscale_cam

grayscale_cam.shape