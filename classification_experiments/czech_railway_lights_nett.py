from torch import nn, flatten

from transformers.modeling_utils import PreTrainedModel
import torch.nn.functional as F
from transformers import PretrainedConfig

from transformers import PretrainedConfig


class CNNConfig(PretrainedConfig):
    model_type = "czech_railway_cnn"

    def __init__(
            self,
            num_labels=6,
            image_size=(16, 34),
            in_channels=3,
            conv_channels=[32, 64],
            kernel_size=(3, 3),
            stride=(1, 1),
            hidden_size=128,
            dropout=0.2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.image_size = image_size
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.hidden_size = hidden_size
        self.dropout = dropout

class CzechRailwayLightNet(PreTrainedModel):
    config_class = CNNConfig
    def __init__(self, config, dp=.2):
        super().__init__(config)
        # Create a small custom model for 16x34 pixel images
        self.config = config
        self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.dropout1 = nn.Dropout(dp)
        self.dropout2 = nn.Dropout(dp)
        self.fc1 = nn.Linear(5760, 512)
        self.fc2 = nn.Linear(512, config.num_labels)

    def forward(self, pixel_values, labels=None):
        x = self.conv1(pixel_values)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        logits = F.log_softmax(x, dim=1)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        # Later, load the model
        loaded_config = CNNConfig(
        num_labels=6,               # Number of railway sign classes
        image_size=(16, 34),        # Input image dimensions
        in_channels=3,              # RGB images
        conv_channels=[32, 64],     # Channels for each conv layer
        kernel_size=(3, 3),         # Kernel size for convolutions
        stride=(1, 1),              # Stride for convolutions
        hidden_size=128,            # Size of FC hidden layer
        dropout=0.2                 # Dropout rate
    )

        loaded_model = CzechRailwayLightNet.from_config(config=loaded_config)
        return loaded_model

    @classmethod
    def from_config(cls, config):
        return cls(config)