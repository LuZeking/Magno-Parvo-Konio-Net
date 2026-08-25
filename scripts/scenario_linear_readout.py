from dataclasses import dataclass


@dataclass
class Params:
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    num_classes: int
    weight_decay: float
    label_smoothing: float
    epsilon: float
    like_resnet: str
    seperate_from_layer: int
    dvd_strategy: str
    age_m: int
    stream_concat: bool
    stream1_weight: float
    lum: bool
    rg: bool
    by: bool
    lum_blur: bool
    lum_cs: bool
    rg_blur: bool
    rg_cs: bool
    by_blur: bool
    by_cs: bool
    combined_color: bool


@dataclass
class Train_Data_Paths:
    log_path: str
    dataset_dir: str
    dataset_name: str
    experiment_name: str


@dataclass
class LinearReadout:
    parameters: Params
    train_data_paths: Train_Data_Paths
