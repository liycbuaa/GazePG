import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dap', help='Choose the model for training:')
    parser.add_argument('--img_size', type=int, default=28, help='Choose the model for training:')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training:')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs:')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate:')
    parser.add_argument('--cuda', type=bool, default=True, help='Use cuda for training ?')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--test_seed', type=int, default=1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=26, help='Number of classes')
    parser.add_argument('--mode', type=str, default='dap')
    parser.add_argument('--source', type=str, default='template26')
    parser.add_argument('--target', type=str, default='gaze26_20_5')
    parser.add_argument('--dataset_root', type=str, default='./datasets')
    parser.add_argument('--model_root', type=str, default='./models')
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--gpu_id', type=str, default='0', help='the gpu device id')

    return check_args(parser.parse_args())


def check_args(args):
    try:
        assert args.num_epochs >= 1
    except:
        print('epoch needs to be larger than one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size needs to be larger than one')

    return args
