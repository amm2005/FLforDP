import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra.utils import instantiate


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


class OneHotEncoding0d(nn.Module):
    """One-hot encodes integer categorical features and concatenates them."""

    def __init__(self, cardinalities):
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x):
        # Clamp guards against stray -1/nan indices causing CUDA out-of-bounds.
        parts = [
            F.one_hot(x[:, i].long().clamp(0, c - 1), c).float()
            for i, c in enumerate(self._cardinalities)
        ]
        return torch.cat(parts, dim=1)


class MLP(nn.Module):
    def __init__(
        self,
        n_num_features,
        cat_cardinalities,
        num_classes,
        hidden_dims,
        dropout=0.0,
    ):
        super().__init__()
        # Fallback: cfg.model updates may not propagate through OmegaConf, so
        # pull authoritative values from the shared preprocessing cache.
        try:
            from utils.preprocessing_cache import preprocessing_cache
            cache = preprocessing_cache
            if cache:
                if not cat_cardinalities:
                    cat_cardinalities = list(cache.get("cat_cardinalities", []))
                num_cols = cache.get("num_cols", [])
                if num_cols and n_num_features != len(num_cols):
                    n_num_features = len(num_cols)
                if "num_classes" in cache:
                    num_classes = cache["num_classes"]
        except Exception:
            pass
        self.cat_module = (
            OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        d_cat = sum(cat_cardinalities) if cat_cardinalities else 0
        d_in = n_num_features + d_cat

        layers = []
        prev_dim = d_in
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x_num=None, x_cat=None):
        parts = []
        if x_num is not None:
            parts.append(x_num)
        if x_cat is not None and self.cat_module is not None:
            parts.append(self.cat_module(x_cat))
        x = torch.cat(parts, dim=1)
        return self.backbone(x)


class BasicModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.model = instantiate(cfg.models[0])

    def forward(self, *args, **kwargs):
        pass


class FLAIRResNet18(nn.Module):
    """ResNet-18 with ImageNet pretraining, head replaced for FLAIR; BN running stats frozen."""

    def __init__(self, num_classes=17, pretrained=True):
        super().__init__()
        # Imported here so torchvision is not a hard dependency at module load.
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(512, num_classes)

    def train(self, mode=True):
        """Set module to training mode, but keep BN layers in eval mode."""
        super().train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        return self.backbone(x)


def flair_resnet18(num_classes=17, pretrained=True):
    """Hydra factory for FLAIRResNet18."""
    return FLAIRResNet18(num_classes=num_classes, pretrained=pretrained)


def _flair_group_norm(num_channels):
    """GroupNorm layer matching the official FLAIR benchmarks."""
    return nn.GroupNorm(num_groups=min(32, num_channels), num_channels=num_channels, eps=1e-5)


class FLAIRResNet18GroupNorm(nn.Module):
    """From-scratch torchvision ResNet-18 with GroupNorm, matching apple/ml-flair."""

    def __init__(self, num_classes=17):
        super().__init__()
        # Imported here so torchvision is not a hard dependency at module load.
        from torchvision.models import resnet18

        self.backbone = resnet18(weights=None, norm_layer=_flair_group_norm)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)


def flair_resnet18_groupnorm(num_classes=17):
    """Hydra factory for FLAIRResNet18GroupNorm."""
    return FLAIRResNet18GroupNorm(num_classes=num_classes)


class BertTiny(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        # Imported here so transformers is an optional dependency at module load.
        from transformers import AutoConfig, AutoModelForSequenceClassification

        self.config = AutoConfig.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
