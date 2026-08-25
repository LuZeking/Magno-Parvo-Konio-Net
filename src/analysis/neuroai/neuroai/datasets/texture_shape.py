from dataclasses import dataclass, field
from os.path import join as pjoin
from typing import List

from .. import constants as c
from ..evaluation import metrics as m
from . import decision_mappings, info_mappings
from .base import Dataset
from .dataloaders import PytorchLoader
from .registry import register_dataset

__all__ = [
    "original",
    "greyscale",
    "texture",
    "edge",
    "silhouette",
    "cue_conflict",
]


@dataclass
class TextureShapeParams:
    path: str
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)])
    decision_mapping: object = (
        decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()
    )
    info_mapping: object = info_mappings.ImageNetInfoMapping()
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False


@dataclass
class TextureShape16Params:
    path: str
    image_size: int = 224
    metrics: list = field(
        default_factory=lambda: [
            m.PresentationSimilarity(),
            m.Accuracy(topk=1),
        ]
    )  # TODO! using m.PresentationSimilarity()
    decision_mapping: object = (
        decision_mappings.ImageNet16ProbabilitiesTo16ClassesMapping()
    )
    info_mapping: object = info_mappings.ImageNetInfoMapping()  # * ???
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False


@dataclass
class MiniecosetTextureShapeParams:
    path: str
    image_size: int = 256
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)])
    decision_mapping: object = (
        decision_mappings.MiniEcosetTexture2ShapeProbabilitiesTo16ClassesMapping()
    )
    info_mapping: object = info_mappings.ImageNetInfoMapping()
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False


@dataclass
class ecosetTextureShapeParams:
    path: str
    image_size: int = 256
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)])
    decision_mapping: object = (
        decision_mappings.ecosetTexture2ShapeProbabilitiesTo16ClassesMapping()
    )
    info_mapping: object = info_mappings.ImageNetInfoMapping()
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False


def _get_dataset(name, *args, **kwargs):
    # Get 'dataset_trained_on' from kwargs if it exists (and remove it from kwargs)
    dataset_trained_on = kwargs.pop("dataset_trained_on", None)

    if dataset_trained_on in ["ecoset", "ecoset_square256"]:
        params = ecosetTextureShapeParams(
            path=pjoin(c.DATASET_DIR, name), image_size=c.IMG_SIZE
        )
    elif dataset_trained_on in [
        "miniecoset_zejin",
        "texture2shape_miniecoset",
    ]:
        params = MiniecosetTextureShapeParams(
            path=pjoin(c.DATASET_DIR, name), image_size=c.IMG_SIZE
        )
    elif dataset_trained_on in ["imagenet"]:
        print("Using default standard imagenet dataset trained model..")
        params = TextureShapeParams(path=pjoin(c.DATASET_DIR, name))
    elif dataset_trained_on == "imagenet16":
        print("Using 16 way imagenet dataset trained model..")
        params = TextureShape16Params(
            path=pjoin(c.DATASET_DIR, name), image_size=c.IMG_SIZE
        )
    else:
        import pdb

        pdb.set_trace()
        raise NotImplementedError
    return Dataset(
        name=name, params=params, loader=PytorchLoader, *args, **kwargs
    )


@register_dataset(name="original")
def original(*args, **kwargs):
    return _get_dataset(name="original", *args, **kwargs)


@register_dataset(name="greyscale")
def greyscale(*args, **kwargs):
    return _get_dataset(name="greyscale", *args, **kwargs)


@register_dataset(name="texture")
def texture(*args, **kwargs):
    return _get_dataset(name="texture", *args, **kwargs)


@register_dataset(name="edge")
def edge(*args, **kwargs):
    return _get_dataset(name="edge", *args, **kwargs)


@register_dataset(name="silhouette")
def silhouette(*args, **kwargs):
    return _get_dataset("silhouette", *args, **kwargs)


@register_dataset(name="cue-conflict")
def cue_conflict(*args, **kwargs):
    return _get_dataset("cue-conflict", *args, **kwargs)
