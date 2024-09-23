# Gaze Pattern Genius: Gaze-Driven VR Interaction using Domain Adaption

![pipelinenew1](image/pipelinenew1.png)

**Abstract:**

In this study, we propose a novel VR paradigm shift towards gaze-driven interaction based on gaze gesture, which offers unique advantages when hands or vocal communication are unavailable. We focus our study on improving the recognition accuracy of neural networks, especially when confronted with the challenge of limited user-specific gaze data. Recognizing the inherent challenges posed by the limited genuine user data for training, we delineate a novel framework for capturing gaze gesture patterns and introduce the concept of a template dataset to enhance neural training. The core of our proposal is a domain adaption model based on Generative Adversarial Networks (GANs). This model fuses the depth of the template dataset with the authenticity of sparse user data, consistently demonstrating an unparalleled adeptness in recognizing gaze gesture patterns across varied user demographics. Our methodology consistently outperforms against leading-edge recognition architectures. To cement the efficacy of our approach, we curate specific VR scenarios and interaction challenges. Empirical results from user studies indicate that gaze-driven interactions not only amplify the VR experience but also herald a new frontier in immersive VR control and interaction dynamics.

## Dataset

The dataset for model training can be downloaded [here](https://drive.google.com/drive/folders/1erGzSPGoFjNOTLkMNK6cB2cgk4agVf_I?usp=sharing). 

Place the dataset the directory specified by `--dataset_root`.

After downloading, the dataset should be organized like this

```
datasets
│  
│──gaze26_20_5
│   │
│   └───trainset
│   └───testset
│    
│──gaze26ForTest
│   │
│   └───trainset
│   └───testset
│
│──template26
│   │
│   └───trainset
│   └───testset
└
```



## Environments

- Python: 3.8
- PyTorch: 2.1.2
- timm: 0.9.2



## Training

```
python main.py --model dap --batch_size 100 --target gaze26_20_5 --mode dap
```

For other models: 

```
python main.py --model vit --batch_size 100 --target gaze26_20_5 --img_size 224 --mode mixture
```



## Acknowledgements

We would like to thank the following repositories for their valuable contributions:

[Mixup_for_UDA](https://github.com/ChrisAllenMing/Mixup_for_UDA), [ViT](https://github.com/huggingface/pytorch-image-models/tree/main), [ConViT](https://github.com/facebookresearch/convit), [MobileViT](https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py), [Swin](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [Regnet](https://github.com/signatrix/regnet), [ResNeXt](https://github.com/facebookresearch/ResNeXt)

