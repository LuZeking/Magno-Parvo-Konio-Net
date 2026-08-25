from collections import OrderedDict

import torch
import torch.nn as nn

CLASSIFIER_WEIGHTS = {
    "InsDis": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/InsDis_classifier.pth",
    "CMC": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/CMC_classifier.pth",
    "MoCo": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/MoCo_classifier.pth",
    "PIRL": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/PIRL_classifier.pth",
    "MoCoV2": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/MoCov2_classifier.pth",
    "InfoMin": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/InfoMin_classifier.pth"
}


PYTORCH_HUB = "kantharajucn/PyContrast:push_pretrained_models_to_pytorch_hub"


def build_classifier(model_name, classes=1000):
    """Building a linear layer"""

    n_class = classes

    n_feat = 2048

    classifier = nn.Linear(n_feat, n_class)
    checkpoint = torch.hub.load_state_dict_from_url(CLASSIFIER_WEIGHTS.get(model_name), map_location='cpu')

    state_dict = OrderedDict()
    for k, v in checkpoint["classifier"].items():
        k = k.replace('module.', '')
        state_dict[k] = v
    classifier.load_state_dict(state_dict)

    return classifier


def InsDis(pretrained=False, **kwargs):
    """
    Unsupervised Feature Learning via Non-parameteric Instance Discrimination
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "InsDis"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)

    return model, classifier


def CMC(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "CMC"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def MoCo(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "MoCo"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def MoCoV2(pretrained=False, **kwargs):
    """
    Improved Baselines with Momentum Contrastive Learning
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "MoCoV2"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def PIRL(pretrained=False, **kwargs):
    """
    Self-Supervised Learning of Pretext-Invariant Representations
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "PIRL"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def InfoMin(pretrained=False, **kwargs):
    """
    What Makes for Good Views for Contrastive Learning?
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "InfoMin"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier

def DINOv2(model_variant='dinov2_vitb14', pretrained=True, classes=1000, **kwargs):
    """
    Load a DINOv2 model and attach a linear classifier.

    Args:
        model_variant (str): Variant of DINOv2 to load. Options include:
            - 'dinov2_vits14'
            - 'dinov2_vitb14'
            - 'dinov2_vitl14'
            - 'dinov2_vitg14'
        pretrained (bool): Whether to load pretrained weights.
        classes (int): Number of output classes for the classifier.

    Returns:
        model (nn.Module): The DINOv2 model.
        classifier (nn.Module): A linear classifier.
    """
    # Load the DINOv2 model from PyTorch Hub
    model = torch.hub.load('facebookresearch/dinov2', model_variant, pretrained=pretrained)

    # Determine the feature dimension based on the model variant
    feature_dims = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }
    n_feat = feature_dims.get(model_variant, 768)  # Default to 768 if variant not found

    # Initialize a linear classifier
    classifier = nn.Linear(n_feat, classes)

    return model, classifier
