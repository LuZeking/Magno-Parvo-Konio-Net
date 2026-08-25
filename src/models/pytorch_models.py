import copy

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101


class TwoStreamNetwork(nn.Module):
    """
    The customizable two stream network.

    Args:
        num_classes (int): Number of output classes (100)
        dropout (float): Value of dropout rate (0.25)
        like_resnet (str): String to determine which resnet arch to use ("resnet34")
        seperate_from_layer (int): The layer at which the two streams seperate (0)
        baseline (bool): Whether to use the baseline architecture (False)
        dvd_strategy (str, optional): The dvd strategy used ("dvd")
        combined_color (bool, optional): Combine the two color channels (False)
        stream_concat (bool, optional): Share information in between the streams in a 3:1 ratio; 3 parts own stream and 1 part other stream (False)
        stream1_weight (float, optional): Indicates the proportion of information coming from stream1 for the classification
    """

    def __init__(
        self,
        num_classes: int = 100,
        dropout: float = 0.25,
        like_resnet: str = "resnet34",
        seperate_from_layer: int = 0,
        baseline: bool = False,
        dvd_strategy: str = "dvd",
        combined_color: bool = False,
        stream_concat: bool = False,
        stream1_weight: float = 0.5,
        legacy_asymmetric_mixing: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dropout = dropout
        self.like_resnet = like_resnet
        self.seperate_from_layer = seperate_from_layer
        self.baseline = baseline
        self.dvd_strategy = dvd_strategy
        self.combined_color = combined_color
        self.stream_concat = stream_concat
        self.stream1_weight = stream1_weight
        # Checkpoints trained before the mixing fix used an asymmetric update
        # (stream 2 received the already-mixed stream-1 tensor, i.e. an
        # effective 0.8125/0.1875 split); set True to reproduce their forward.
        self.legacy_asymmetric_mixing = legacy_asymmetric_mixing

        self.setup_model_blocks()

    def setup_model_blocks(self):
        "Create the building blocks of the two stream model"
        if self.like_resnet not in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
        ]:
            raise ValueError(
                f"Invalid like_resnet value: {self.like_resnet}. Must be one of 'resnet18', 'resnet34', 'resnet50', or 'resnet101'."
            )

        if self.baseline and self.seperate_from_layer != 0:
            raise ValueError(
                "Baseline architecture cannot have seperate_from_layer greater than 0."
            )

        if self.baseline and self.stream_concat:
            raise ValueError(
                "The Baseline architecture does not have two streams to concatinate the activations with."
            )

        if self.stream_concat and self.seperate_from_layer != 0:
            raise ValueError(
                "seperate_from_layer cannot be greater than 0 if stream_concat is set to true."
            )

        if self.dvd_strategy in ["dkl", "dkl_dvd"] and self.seperate_from_layer != 0:
            raise ValueError("A shared block and DKL input is not supported.")

        shared_layers = []
        individual_layers_block1 = []
        individual_layers_block2 = []
        classifier_layers = []

        models = {
            "resnet18": resnet18(weights=None),
            "resnet34": resnet34(weights=None),
            "resnet50": resnet50(weights=None),
            "resnet101": resnet101(weights=None),
        }

        resnet = models[self.like_resnet]

        num_layers = len(list(resnet.children()))

        # layer allocation
        for i, layer in enumerate(resnet.children()):

            # include some layers in a shared block
            if i < self.seperate_from_layer:
                shared_layers.append(layer)

            # if stream_concat is true, cut the resnet streams after two blocks and concatinate the outputs in a 3:1 ratio
            elif self.stream_concat:
                if i < (num_layers - 4):
                    # when using several strategies the input only has one channel per stream, hence we need to update the number of input channels
                    if i == 0 and self.dvd_strategy in ["dkl", "dkl_dvd"]:

                        layer = nn.Conv2d(
                            1,
                            64,
                            kernel_size=(7, 7),
                            stride=(2, 2),
                            padding=(3, 3),
                            bias=False,
                        )
                    individual_layers_block1.append(layer)

                elif i < (num_layers - 2):

                    individual_layers_block2.append(layer)

                else:
                    if layer.__class__.__name__ == "Linear":
                        classifier_layers.append(nn.Flatten())
                        layer = nn.Linear(layer.in_features, self.num_classes)
                    classifier_layers.append(layer)

            else:
                if i < (num_layers - 2):
                    # when using several strategies the input only has one channel per stream, hence we need to update the number of input channels
                    if i == 0 and self.dvd_strategy in ["dkl", "dkl_dvd"]:
                        if self.combined_color and self.baseline:
                            in_channels = 2
                        else:
                            in_channels = 1

                        layer = nn.Conv2d(
                            in_channels,
                            64,
                            kernel_size=(7, 7),
                            stride=(2, 2),
                            padding=(3, 3),
                            bias=False,
                        )
                    individual_layers_block1.append(layer)
                else:
                    if layer.__class__.__name__ == "Linear":
                        classifier_layers.append(nn.Flatten())
                        layer = nn.Linear(layer.in_features, self.num_classes)
                    classifier_layers.append(layer)

        # stream creation
        if len(shared_layers) == 0:
            self.shared_block = None
        else:
            self.shared_block = copy.deepcopy(nn.Sequential(*shared_layers))

        if self.stream_concat:
            self.stream1_block1 = copy.deepcopy(
                nn.Sequential(*individual_layers_block1)
            )
            self.stream1_block2 = copy.deepcopy(
                nn.Sequential(*individual_layers_block2)
            )

            # since the combine the colors the number of input channels needs to be 2
            if self.combined_color:
                individual_layers_block1[0] = nn.Conv2d(
                    2,
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )

            self.stream2_block1 = copy.deepcopy(
                nn.Sequential(*individual_layers_block1)
            )
            self.stream2_block2 = copy.deepcopy(
                nn.Sequential(*individual_layers_block2)
            )

        else:
            self.stream1_block1 = copy.deepcopy(
                nn.Sequential(*individual_layers_block1)
            )
            self.stream1_block2 = None

            if self.baseline:
                self.stream2_block1 = None
                self.stream2_block2 = None
            else:
                if self.combined_color:
                    individual_layers_block1[0] = nn.Conv2d(
                        2,
                        64,
                        kernel_size=(7, 7),
                        stride=(2, 2),
                        padding=(3, 3),
                        bias=False,
                    )
                self.stream2_block1 = copy.deepcopy(
                    nn.Sequential(*individual_layers_block1)
                )

        self.classifier = copy.deepcopy(nn.Sequential(*classifier_layers))

    def _mix_streams(self, y1, y2):
        """Cross-stream mixing after block 2 (75% own stream, 25% other).

        The legacy variant fed the already-mixed y1 into y2's update, so
        stream 2 effectively received 0.8125*y2 + 0.1875*y1.
        """
        if self.legacy_asymmetric_mixing:
            y1 = (3 * y1 + y2) / 4
            y2 = (3 * y2 + y1) / 4
            return y1, y2
        y1_pre, y2_pre = y1, y2
        y1 = (3 * y1_pre + y2_pre) / 4
        y2 = (3 * y2_pre + y1_pre) / 4
        return y1, y2

    def forward(self, x_1, x_2=None):
        "Forward pass"

        if x_2 is None:
            # one stream baseline
            if self.baseline:
                x_1 = self.stream1_block1(x_1)
                y_hat = self.classifier(x_1)

            # two stream with the same input
            else:
                # after two blocks of the resnet concatinate the streams in a 3:1 ratio
                if self.stream_concat:
                    y1 = self.stream1_block1(x_1)
                    y2 = self.stream2_block1(x_1)

                    y1, y2 = self._mix_streams(y1, y2)

                    y1 = self.stream1_block2(y1)
                    y2 = self.stream2_block2(y2)

                    y_comb = (self.stream1_weight * y1) + (
                        (1 - self.stream1_weight) * y2
                    )
                    y_hat = self.classifier(y_comb)

                # two stream forward pass with the same input and optionally a shared block in the beginning
                else:
                    if self.shared_block is not None:
                        x_1 = self.shared_block(x_1)
                    y1 = self.stream1_block1(x_1)
                    y2 = self.stream2_block1(x_1)
                    y_comb = (self.stream1_weight * y1) + (
                        (1 - self.stream1_weight) * y2
                    )
                    y_hat = self.classifier(y_comb)

        # two stream with different input
        else:
            # raise error if the architecture doesnt support two inputs
            if self.shared_block is not None or self.baseline:
                raise ValueError(
                    "You cannot give the model two different inputs when the architecture just has a single stream input."
                )
            if self.stream_concat:
                y1 = self.stream1_block1(x_1)
                y2 = self.stream2_block1(x_2)

                y1, y2 = self._mix_streams(y1, y2)

                y1 = self.stream1_block2(y1)
                y2 = self.stream2_block2(y2)

                y_comb = (self.stream1_weight * y1) + (
                    (1 - self.stream1_weight) * y2
                )
                y_hat = self.classifier(y_comb)

            else:
                y1 = self.stream1_block1(x_1)
                y2 = self.stream2_block1(x_2)
                y_comb = (self.stream1_weight * y1) + (
                    (1 - self.stream1_weight) * y2
                )
                y_hat = self.classifier(y_comb)

        return y_hat


# network = TwoStreamNetwork(like_resnet="resnet18", baseline=False, seperate_from_layer=0, stream_concat=True, dvd_strategy="dkl", combined_color=False)
# trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

# print(network)
# print(trainable_params)


class LinearReadoutModel(nn.Module):
    "The fully connected model for linear readouts"

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 112,
    ):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.setup_model()

    def setup_model(self):
        "Create the fully connected layer"

        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(self.input_dim, self.output_dim)
        )

    def forward(self, x):
        "Forward pass"

        y_hat = self.fc(x)

        return y_hat
