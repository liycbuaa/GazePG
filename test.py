import argparse

import torch

import data_loader
from model import build_model


def testing(args, model_path):
    model = build_model(args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    target_test_loader = data_loader.load_testing(args.img_size, "./datasets/", "gaze26ForTest", 128)
    with torch.no_grad():
        correct = 0
        for idx, (data, label) in enumerate(target_test_loader):
            if args.model == 'revgrad':
                _, pred, _ = model(data, alpha=0)
            else:
                pred = model(data)
            # test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, size_average=False).item()
            pred = pred.max(1)[1]
            # print(pred)
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            # for id,label in enumerate(pred):
            #     plt.subplot(8, 8, id+1)
            #     plt.imshow(data[id].permute(1, 2, 0))
            #     plt.title(chr(ord('A')+label))
            #     plt.axis('off')
            #     plt.tight_layout()
            #     plt.savefig("../models/result/re"+str(idx)+'.jpg')

        len_target_dataset = len(target_test_loader.dataset)
        acc = 100. * correct / len_target_dataset

    print('gaze26ForTest Accuracy : {:.2f}%'.format(acc))

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dap', help='Choose the model for training:')
    parser.add_argument('--num_classes', type=int, default=26, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=28, help='Choose the model for training:')
    model_path = 'tgtloss_best_modelgaze26_20_5.pkl'
    args = parser.parse_args()
    testing(args, model_path)
