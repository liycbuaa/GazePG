import logging

from config import parse_args
from test import testing
from trainer.trainer import Trainer
from trainer.trainer_dap import TrainerDAP

if __name__ == '__main__':
    # args
    args = parse_args()
    # Trainer
    if args.model == 'dap':
        trainer = TrainerDAP(args)
    else:
        trainer = Trainer(args)
    model_path = trainer.train(args)
    # test
    acc = testing(args, model_path=model_path)
    logging.info('gaze26ForTest Accuracy : {:.2f}%'.format(acc))
