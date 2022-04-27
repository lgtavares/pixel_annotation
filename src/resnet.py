import torch
import torchvision.models as models
from PIL import Image

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]

class Resnet50():

    def __init__(self, device=None, pretrained=True):
        self.resnet50 = models.resnet50(pretrained=pretrained)
        if device is None:
            device = 'cpu'
        self._send_model_to_device(device)

        self.transformations = None
        self.hooks = {}

        # Com hook: residual3
        self._layers_points = {'residual3': {'layer_name': 'layer1', 'id_module': '2'}}

    def __call__(self, frame_RGB):
        # Transform image and pass it throught the network
        image = Image.fromarray(frame_RGB)
        t_img = self.transformations(image).float().to(self.device).unsqueeze(0)
        return self.resnet50(t_img)

    def _send_model_to_device(self, device):
        if type(device) is str:
            device = device.lower()
            assert device in ['cuda', 'cpu']
            self.device = torch.device(device)
        elif type(device) is torch.device:
            self.device = device
        self.resnet50.to(self.device)

    def _add_hook(self, layer_name, callback, parameters_callback):
        if self._layers_points[layer_name]['id_module'] is None:
            hook = self.resnet50._modules.get(layer_name)
        else:
            hook = self.resnet50._modules.get(
                self._layers_points[layer_name]['layer_name'])._modules.get(
                    self._layers_points[layer_name]['id_module'])
        self.hooks[layer_name] = hook.register_forward_hook(callback(parameters_callback))

    def _remove_all_hooks(self):
        for hook in self.hooks:
            self.hooks[hook].remove()
        self.hooks.clear()

    def get_features(self, image, layer_to_extract):
        def features_extracted_callback(parameters):
            def hook(m, i, o):
                assert parameters['layer_name'] == layer_to_extract
                self._features_extracted = o.data.squeeze()

            return hook

        if layer_to_extract not in self.hooks:
            self._add_hook(layer_to_extract, features_extracted_callback,
                           {'layer_name': layer_to_extract})
        
        t_img = image.float().to(self.device)

        self.resnet50(t_img)
        return self._features_extracted

        
class Resnet50_Reduced():

    MEAN_IMAGENET = [0.485, 0.456, 0.406]
    STD_IMAGENET = [0.229, 0.224, 0.225]

    def __init__(self, device=None, pretrained=True, learning=False):
        self.device = device

        if device is None:
            device = 'cpu'

        # Corta a resnet50 para ficar somente com as camadas iniciais (até a residual3).
        self.resnet50 = torch.nn.Sequential(
            *list(models.resnet50(pretrained=pretrained).children())[0:5])
        if learning:
            self.resnet50.train()
        else:
            self.resnet50.eval()

        self.resnet50 = self.resnet50.to(self.device)

    def send_to_device(self, device):
        self.device = device
        self.resnet50 = self.resnet50.to(self.device)

    def freeze(self):
        for p in self.resnet50.parameters():
            p.requires_grad = False

    def __call__(self, frame_RGB):
        # image = Image.fromarray(frame_RGB.float().to(self.device))
        # t_img = self.transformations(image).float().to(self.device).unsqueeze(0)
        if frame_RGB.device == self.resnet50:
            return self.resnet50(frame_RGB)
        else:
            if frame_RGB.device.type != 'cuda':
                return self.resnet50(frame_RGB.to(self.device))
            else:
                return self.resnet50(frame_RGB)
