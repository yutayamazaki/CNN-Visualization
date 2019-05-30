import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import config


class GuidedBackpropagation:

    def __init__(self, model, img_size=config.IMG_SIZE, transform=None, use_cuda=False):
        self.model = model.eval()
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

        for module in self.model.named_modules():
            module[1].register_backward_hook(self._relu_backward)

    def _relu_backward(self, module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, x, index=None):
        if self.use_cuda:
            x = x.cuda()
        x.requires_grad_()
        output = self.model(x)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad_()

        if self.use_cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward()
        gbp_img = x.grad.cpu().numpy()[0]
        gbp_img = np.transpose(gbp_img, (1, 2, 0))  # [C, H, W] -> [H, W, C]

        gbp_img = np.maximum(gbp_img, 0)
        gbp_img = gbp_img - np.min(gbp_img)
        gbp_img = gbp_img / np.max(gbp_img)

        return gbp_img

    def prepare_torch_input(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path)
        return self.transform(img).unsqueeze(0)


if __name__ == '__main__':
    gbp_model = GuidedBackpropagation(
        model=torchvision.models.resnet34(pretrained=True),
        use_cuda=True
    )
    img_path = os.path.join(config.IMG_DIR, config.SAMPLE_IMG)
    img = gbp_model.prepare_torch_input(img_path)
    guided_img = gbp_model(img)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(Image.open(img_path).resize((config.IMG_SIZE, config.IMG_SIZE)))

    save_img_path = os.path.join(config.IMG_DIR, 'guided_backprop.png')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(guided_img)
    plt.savefig(save_img_path)
    plt.show()