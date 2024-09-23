from torch import nn

from model.ConVit import convit_tiny as create_convit
from model.ConvNext import convnext_tiny as create_convnext
from model.MobileVit import mobile_vit_xx_small as create_mobilevit
from model.RegNet import create_regnet
from model.ResNeXt import resnet34
from model.SwinTransformer import swin_tiny_patch4_window7_224 as create_swin
from model.vit import vit_base_patch16_224 as create_vit
from model.Classifier import Classifier
from model.RevGrad import RevGrad

def build_model(args):
    if args.model == 'dap':
        model = RevGrad(num_classes=args.num_classes)
    elif args.model == 'regnet':
        model = create_regnet(num_classes=args.num_classes)
    elif args.model == 'resnext':
        model = resnet34(num_classes=args.num_classes)
        in_channel = model.fc.in_features
        model.fc = nn.Linear(in_channel, args.num_classes)
    elif args.model == 'vit':
        model = create_vit(num_classes=args.num_classes)
    elif args.model == 'swin':
        model = create_swin(num_classes=args.num_classes)
    elif args.model == 'convnext':
        model = create_convnext(num_classes=args.num_classes)
    elif args.model == 'mobilevit':
        model = create_mobilevit(num_classes=args.num_classes)
    elif args.model == 'convit':
        model = create_convit(
            img_size=64,
            num_classes=args.num_classes,
            embed_dim=48
        )
    elif args.model == 'classifier':
        model = Classifier(num_classes=args.num_classes)
    else:
        raise Exception('Model is not defined!')
    return model
