from torch import nn
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel

class CzechRailwayLightNet(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Create a small custom model for 16x34 pixel images
        self.config = config

        # Simple CNN architecture suitable for small images
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Calculate the flattened size after convolutions
        # After 2 max pooling layers with stride 2, the 16x34 becomes 4x8
        self.avgpool = nn.AdaptiveAvgPool2d((4, 8))
        self.flatten_size = 128 * 4 * 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, config.num_labels)
        )

    def forward(self, pixel_values, labels=None):
        # Process the input images
        features = self.features(pixel_values)
        features = self.avgpool(features)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, num_labels=None, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
        model = cls(config)
        return model

    @classmethod
    def from_config(cls, config):
        return cls(config)
