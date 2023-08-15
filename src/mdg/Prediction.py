import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import mdg

log = mdg.utils.get_logger()


class Prediction:

    def __init__(self, testset, model, args):
        self.testset = testset
        self.model = model
        self.args = args
        self.best_dev_ccc = None
        self.best_test_ccc = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_ccc = ckpt["best_dev_ccc"]
        self.best_test_ccc = ckpt["best_test_ccc"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def pred(self):
        
        dataset = self.testset
        self.model.eval()
        with torch.no_grad():
            gt = []
            preds = []
            for idx in tqdm(range(len(dataset))):
                data = dataset[idx]
                gt.append(torch.nan_to_num(data["label_tensor"], nan=0.))
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(torch.nan_to_num(y_hat, nan=0.))

            gt = torch.concat(gt, 0).to(self.args.device)
            preds = torch.concat(preds, 0).to(self.args.device)
            gt = torch.reshape(gt, (-1,)).detach().to("cpu").numpy()
            preds = torch.reshape(preds, (-1,)).detach().to("cpu").numpy()
        return gt, preds
