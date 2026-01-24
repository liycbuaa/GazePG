import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.autograd import Function
from torch.autograd import Variable
import torchvision.models as models
import math
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoModel

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class RevGrad(nn.Module):

    def __init__(self, num_classes=10, dim=384):
        super(RevGrad, self).__init__()
        self.nclasses = num_classes
        self.out_dim = 256

        # 基于 dinov2 的 small vit
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-small')
        # lora 设置，秩数16，添加在注意力层的query和value
        self.config = LoraConfig(
            r=16,  # 类似reduction_factor
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        # 通过huggingface peft包的api构造基于lora微调的vit
        self.feature = get_peft_model(self.backbone, self.config)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(dim, self.out_dim))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(self.out_dim))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(self.out_dim, self.out_dim))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(self.out_dim))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(self.out_dim, self.nclasses))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(dim, self.out_dim))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(self.out_dim))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.out_dim, 2))

    def forward(self, input_data, alpha):
        outputs = self.feature(input_data)
        feature = outputs.last_hidden_state[:, 0]
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return feature, class_output, domain_output
