import timm
import torch.nn as nn

MODEL_NAMES = [
    'baseline_cnn',
    'resnet18',
    'resnet34',
    'resnet50',
    'densenet121',
    'efficientnet_b0',
]

CLASSIFIER_PREFIXES = ['classifier', 'fc', 'head', 'linear', 'last_linear']


def build_baseline_cnn(num_classes: int):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )
    return model


def build_model(name: str, num_classes: int, pretrained: bool = True):
    name = name.lower()
    if name == 'baseline_cnn':
        return build_baseline_cnn(num_classes)

    if name in MODEL_NAMES:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f'Unsupported model: {name}')


def _unfreeze_classifier(model: nn.Module) -> None:
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
        return
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
        return
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
        return

    for name, param in model.named_parameters():
        if any(prefix in name.lower() for prefix in CLASSIFIER_PREFIXES):
            param.requires_grad = True


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    _unfreeze_classifier(model)


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def is_transfer_learning_model(name: str) -> bool:
    return name.lower() != 'baseline_cnn'
