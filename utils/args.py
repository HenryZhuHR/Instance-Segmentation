from email.policy import default
import os
import argparse
from typing import List
import torch


class ARGSer:
    def __init__(self) -> None:
        args = self.init_args()
        self.args = args

        # ====== Dataset ======
        self.DATASET: str = args.dataset
        self.DATAROOT: str = args.data_root

        # ====== Model Architecture ======
        self.MODEL: str = args.model
        self.FEATURE_NORM: bool = args.feature_norm

        # ====== Model Weight ======
        self.MODEL_PATH: str = args.model_path
        self.RESUME: str = args.resume
        self.LAST_EPOCH: str = args.last_epoch

        # ====== Train settings ======
        self.DEVICE: str = torch.device(args.device)
        self.TRAIN_BATCH_SIZE: int = args.train_batch_size
        self.VALID_BATCH_SIZE: int = args.valid_batch_size
        self.EPOCHS: int = args.epochs        
        self.NUM_WORKERS: int = args.num_workers
        self.CLIP_GRID: float = args.clip_grad
        self.WEIGHT_DECAY: float = args.weight_decay

        # ====== LR schedule ======
        self.WARM_UP: int = args.warm_up
        self.LR: float = args.lr
        self.LR_MIN: float = args.lr_min
        self.GAMMA: float = args.gamma
        self.SCHEDULER: str = args.scheduler
        self.SCHEDULE: List[int] = args.schedule

        # ====== Save ======
        self.SAVE_NAME: str = args.save_name
        self.SAVE_DIR: str = args.save_dir

    def init_args(self):
        parser = argparse.ArgumentParser()
        # ====== Dataset ======
        parser.add_argument('--dataset', type=str)
        DEFAULT_DATAROOT = os.path.expandvars('$HOME/datasets')
        parser.add_argument('--data_root', type=str, default=DEFAULT_DATAROOT)

        # ====== Model Architecture ======
        parser.add_argument('--model', type=str, default='resnet18')
        parser.add_argument('-fn', '--feature_norm', action="store_true")

        # ====== Model Weight =======
        parser.add_argument('--model_path', type=str)
        parser.add_argument('--resume', type=str)
        parser.add_argument('--last_epoch', type=int)

        # ====== Train settings ======
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--train_batch_size', type=int, default=4)
        parser.add_argument('--valid_batch_size', type=int, default=4)
        parser.add_argument('--epochs', type=int, default=300)        
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--clip_grad', type=float)
        parser.add_argument('--weight_decay', type=float, default=5e-4)

        parser.add_argument('--seed', type=int, default=1024)


        # ====== LR schedule ======
        parser.add_argument('--warm_up', type=int, default=0)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--lr_min', type=float, default=0.001)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--scheduler', type=str, default='step',
                            choices=['step', 'cos','exp'])
        parser.add_argument('--schedule', type=int,
                            nargs='+',  default=[150, 225])

        # ====== Save ======
        parser.add_argument('--save_dir', type=str, default='tmp')
        parser.add_argument('--save_name', type=str, default='test')

        args = parser.parse_args()

        SEED = args.seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # np.random.seed(SEED)
        # random.seed(SEED)
        torch.backends.cudnn.deterministic = True

        print(chr(128640), '\033[01;36m', args, '\033[0m')
        return args
