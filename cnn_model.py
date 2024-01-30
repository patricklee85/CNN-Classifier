
import os
import torch
import torchvision
from utils.torch_device import get_device
from efficientnet_pytorch import EfficientNet
from models.resnet import resnet50
from models.resnet import resnet34

def set_parameter_requires_grad(model, requires_grad):
    for i, param in enumerate(model.parameters()):
            param.requires_grad = requires_grad


def freeze_layer(model, freeze_count=2):
    i = 0
    for child in model.children():
        try:
            # iterable
            iterator = iter(child)
            for subchild in iterator:
                if i < freeze_count:
                    for param in subchild.parameters():
                        param.requires_grad = False
                    print(f'[freeze] layer, {i}, {child._get_name()}, {subchild._get_name()}')
                    i += 1
                else:
                    print(f'layer, {i}, {child._get_name()}, {subchild._get_name()}')
        except TypeError:
            # not iterable
            if i < freeze_count:
                for param in child.parameters():
                    param.requires_grad = False
                print(f'[freeze] layer, {i}, {child._get_name()}')
                i += 1
            else:
                print(f'layer, {i}, {child._get_name()}, {subchild._get_name()}')

def initialize_resnet34(num_classes, load_model=''):

    # model = torchvision.models.resnet50(True)
    model = resnet34(True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, num_classes))

    for name, module in model.named_modules():
        for param in module.parameters():
            param.requires_grad = True
    freeze_layer(model, 4)
    
    if os.path.isfile(load_model) and os.path.exists(load_model):
        model.load_state_dict(
            torch.load(
                load_model, 
                map_location = get_device()))

    return model

def initialize_resnet50(num_classes, load_model=''):

    # model = torchvision.models.resnet50(True)
    model = resnet50(True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, num_classes))

    for name, module in model.named_modules():
        for param in module.parameters():
            param.requires_grad = True
    freeze_layer(model, 5)
    
    if os.path.isfile(load_model) and os.path.exists(load_model):
        model.load_state_dict(
            torch.load(
                load_model, 
                map_location = get_device()))

    return model


def initialize_resnet101(num_classes, load_model='', use_finetune=False):

    model = torchvision.models.resnet101(pretrained=True)
    if use_finetune:
        set_parameter_requires_grad(model, False)
    else:
        set_parameter_requires_grad(model, True)
        freeze_layer(model, 7)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, num_classes))
    
    if os.path.isfile(load_model) and os.path.exists(load_model):
        model.load_state_dict(
            torch.load(
                load_model, 
                map_location = get_device()))
    
    return model


def initialize_efficientnet_b0(num_classes, load_model='', use_finetune=False):

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    model.set_swish(memory_efficient=False)
    
    for param in model.parameters():
        param.requires_grad = False

    enable_requires_grad = False
    requires_grad_hit_layer = '_fc' if use_finetune else '_blocks.13'
    for name, module in model.named_modules():
        # print(name)
        if  name == requires_grad_hit_layer:
            enable_requires_grad = True
            
        if  enable_requires_grad:
            for param in module.parameters():
                param.requires_grad = True

    if os.path.isfile(load_model) and os.path.exists(load_model):
        model.load_state_dict(
            torch.load(
                load_model, 
                map_location = get_device()))
    
    return model


def initialize_efficientnet_b3(num_classes, load_model='', use_finetune=False):

    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
    model.set_swish(memory_efficient=False)
    
    for param in model.parameters():
        param.requires_grad = False

    enable_requires_grad = False
    requires_grad_hit_layer = '_fc' if use_finetune else '_blocks.23'
    for name, module in model.named_modules():
        # print(name)
        if  name == requires_grad_hit_layer:
            enable_requires_grad = True
            
        if  enable_requires_grad:
            for param in module.parameters():
                param.requires_grad = True

    if os.path.isfile(load_model) and os.path.exists(load_model):
        model.load_state_dict(
            torch.load(
                load_model, 
                map_location = get_device()))
    
    return model


def initialize_efficientnet_b4(num_classes, load_model='', use_finetune=False):

    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    model.set_swish(memory_efficient=False)
    
    for param in model.parameters():
        param.requires_grad = False

    enable_requires_grad = False
    requires_grad_hit_layer = '_fc' if use_finetune else '_blocks.29'
    for name, module in model.named_modules():
        # print(name)
        if  name == requires_grad_hit_layer:
            enable_requires_grad = True
            
        if  enable_requires_grad:
            for param in module.parameters():
                param.requires_grad = True

    if os.path.isfile(load_model) and os.path.exists(load_model):
        model.load_state_dict(
            torch.load(
                load_model, 
                map_location = get_device()))
    
    return model


def initialize_yolov5_classifier(num_classes, load_model='', use_finetune=False):
    from utils.torch_utils import intersect_dicts
    from models.yolo import Model

    # args
    anchors = 3
    freeze = 23 if use_finetune else 10
    device = get_device()
    weights = r'D:\BackendAOI\Python\Yolov5\yolov5\weights\yolov5s.pt'
    cfg = r'D:\BackendAOI\Python\Yolov5\yolov5\models\yolov5s.yaml'
    
    # load model weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=num_classes, anchors=anchors).to(device)  # create
        exclude = ['anchor'] if (cfg or anchors) else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
    else:
        model = Model(cfg, ch=3, nc=num_classes, anchors=anchors).to(device)  # create

    for k, v in model.named_parameters():
        print(k)

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # load model
    if os.path.isfile(load_model) and os.path.exists(load_model):
        model.load_state_dict(
            torch.load(
                load_model, 
                map_location = get_device()))

    return model


# model = initialize_yolov5_classifier(2)
# x = torch.rand((1,3,640,640))
# x = x.to(get_device())
# print(model(x))