import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torchvision

import config


class FeatureExtractor:

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name == 'fc':
                continue
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(
            model=self.model,
            target_layers=target_layers
        )

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.fc(output)
        return target_activations, output


class GradCam:

    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, inputs):
        return self.model(inputs)

    def __call__(self, inputs, index=None):
        if self.cuda:
            features, output = self.extractor(inputs.cuda())
        else:
            features, output = self.extractor(inputs)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # (B, C, H, W)
        grads = self.extractor.get_gradients()[-1].cpu().data.numpy()

        # (B, C, H, W) -> (C, H, W)
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        # (C, )
        weights = np.mean(grads, axis=(2, 3))[0, :]
        # (H, W)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, alpha in enumerate(weights):
            cam += alpha * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    # img from cv2 BGR
    img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img)
    img_tensor.unsqueeze_(0)
    return img_tensor


def concat_mask(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    gcmodel = GradCam(model=torchvision.models.resnet18(True),
                      target_layer_names=['layer4'],
                      use_cuda=False)

    img_path = os.path.join(config.IMG_DIR, config.SAMPLE_IMG)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = np.float32(img) / 255.

    x = img_to_tensor(img)

    mask = gcmodel(x)
    gradcam_img = concat_mask(img, mask)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(img[:, :, ::-1])

    save_img_path = os.path.join(config.IMG_DIR, 'gradcam.png')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(gradcam_img[:, :, ::-1])
    plt.savefig(save_img_path)
    plt.show()