# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:37

@author: mick.yi

"""
import numpy as np
import cv2
import torch


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, reshape_transform=None):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.reshape_transform = reshape_transform
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        if self.reshape_transform is not None:
            output = self.reshape_transform(output)
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        if isinstance(self.layer_name, str):
            for name, module in self.net.named_modules():
                if name == self.layer_name:
                    self.handlers.append(module.register_forward_hook(self._get_features_hook))
                    self.handlers.append(module.register_backward_hook(self._get_grads_hook))
                    break
        elif isinstance(self.layer_name, list):  # list of modules
            for module in self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))
        elif isinstance(self.layer_name, torch.nn.Module):
            self.handlers.append(self.layer_name.register_forward_hook(self._get_features_hook))
            self.handlers.append(self.layer_name.register_backward_hook(self._get_grads_hook))
        else:
            raise ValueError("Invalid layer_name type.")

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        # print(f"inputs:{inputs.shape}")
        output = self.net(inputs.float())  # [1,num_classes]
        if isinstance(output, tuple):
            output = output[0]
        # print(f"output:{output.shape}")
        # if index is None:
        #     index = np.argmax(output.cpu().data.numpy())
        if index is None:
            index = 0
        target = output[0][index]
        target.backward(retain_graph=True)

        if self.gradient is None:
            raise RuntimeError("Gradient is None. Check if the layer is correct and hooks are triggered.")

        gradient = self.gradient
        if self.reshape_transform is not None:
            gradient = self.reshape_transform(gradient)
        gradient = gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name, reshape_transform=None):
        super(GradCamPlusPlus, self).__init__(net, layer_name, reshape_transform)

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs.float())  # [1,num_classes]
        if isinstance(output, tuple):
            output = output[0]
        # if index is None:
        #     index = np.argmax(output.cpu().data.numpy())
        if index is None:
            index = 0
        target = output[0][index]
        target.backward(retain_graph=True)

        if self.gradient is None:
            raise RuntimeError("Gradient is None. Check if the layer is correct and hooks are triggered.")

        gradient = self.gradient
        if self.reshape_transform is not None:
            gradient = self.reshape_transform(gradient)
        gradient = gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam
