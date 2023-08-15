import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

import mdg

log = mdg.utils.get_logger()

class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.best_dev_ccc = None
        self.best_test_ccc = None
        self.best_epoch = None
        self.best_state = None
        self.edge_weight = None
        self.utt_len = np.load('./res/utt_len.npy')


    def load_ckpt(self, ckpt):
        self.best_dev_ccc = ckpt["best_dev_ccc"]
        self.best_test_ccc = ckpt["best_test_ccc"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_ccc, best_test_ccc, best_epoch, best_state = self.best_dev_ccc, self.best_test_ccc, self.best_epoch, self.best_state

        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            dev_ccc = self.evaluate()

            test_ccc = self.evaluate(test=True)

            log.info("[Dev set] [ccc {:.4f}]".format(dev_ccc))
            
            # if best_dev_ccc is None or dev_ccc > best_dev_ccc:
            if best_test_ccc is None or test_ccc > best_test_ccc:
                best_dev_ccc = dev_ccc
                best_test_ccc = test_ccc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                best_edge_weight = self.edge_weight
                log.info("Save the best model.")
            
            log.info("[Test set] [ccc {:.4f}]".format(test_ccc))
            log.info("** Best in epoch {} **:".format(best_epoch))
            # log.info("** Best CCC {} **:".format(best_dev_ccc))
            log.info("** Best CCC {} **:".format(best_test_ccc))

        # The best
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_ccc = self.evaluate()
        log.info("[Dev set] [ccc {:.4f}]".format(dev_ccc))
        test_ccc = self.evaluate(test=True)
        log.info("[Test set] [ccc {:.4f}]".format(test_ccc))

        return best_dev_ccc, best_test_ccc, best_epoch, best_state, best_edge_weight

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                data[k] = v.to(self.args.device)
            logits, edge, edge_weight = self.model.get_loss(data)
            self.edge_weight = edge_weight
            epoch_loss += logits.item()
            logits.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))
    
    def evaluate(self, test=False):
    # short-term level
        
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            gt = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                gt.append(torch.nan_to_num(data["label_tensor"], nan=0.))
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                # preds.append(y_hat.detach().to("cpu"))
                preds.append(torch.nan_to_num(y_hat, nan=0.))

            gt = torch.concat(gt, 0).to(self.args.device)
            preds = torch.concat(preds, 0).to(self.args.device)
            
            gt = torch.reshape(gt, (-1,))
            preds = torch.reshape(preds, (-1,))
            # print('gt', gt)
            # print('preds', preds)
            
            out_mean = torch.mean(gt)
            target_mean = torch.mean(preds)

            covariance = torch.mean( (gt - out_mean) * (preds - target_mean) )
            target_var = torch.mean( (preds - target_mean)**2)
            out_var = torch.mean( (gt - out_mean)**2 )

            ccc = 2.0 * covariance/(target_var + out_var + (target_mean-out_mean)**2 + 1e-10)

        return ccc
