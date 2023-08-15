import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mdg

log = mdg.utils.get_logger()


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Predictor, self).__init__()
        self.emotion_att = MaskedSegAtt(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)
        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()
        self.utt_len = np.load('./res/utt_len.npy')
        self.device = args.device

    def get_score(self, h, text_len_tensor):
        h_hat = self.emotion_att(h, text_len_tensor)
        hidden = self.drop(F.relu(self.lin1(h_hat)))
        scores = self.relu(self.lin2(hidden))
        
        return scores

    def forward(self, h, text_len_tensor):
        log_prob = self.get_score(h, text_len_tensor)

        return log_prob

    def CCCLoss(self, x, y):
        x_utt = []
        y_utt = []

        y = torch.reshape(y, (-1,))
        x = torch.reshape(x, (-1,))

        for i in range(len(self.utt_len)-1):
            y_utt.append(y[self.utt_len[i]+1])
            st, et = self.utt_len[i]+1, self.utt_len[i]+1+self.utt_len[i+1]
            x_utt.append(torch.mean(x[st:et]))

        x_utt = torch.tensor(x_utt).to(self.device)
        y_utt = torch.tensor(y_utt).to(self.device)

        ccc = 2*torch.cov(torch.concat([x_utt, y_utt], 0)) / (x_utt.var() + y_utt.var() + (x_utt.mean() - y_utt.mean())**2)

        return 1-ccc

    def get_loss(self, h, label_tensor, text_len_tensor):
        label_tensor = torch.reshape(label_tensor, (-1, 1))
        score_pred = self.get_score(h, text_len_tensor)
        loss1 = self.mse_loss(score_pred, label_tensor)
        loss2 = self.CCCLoss(score_pred, label_tensor)
        return loss1 + loss2


class MaskedSegAtt(nn.Module):

    def __init__(self, input_dim):
        super(MaskedSegAtt, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim)

    def forward(self, h, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        x = self.lin(h)  # [node_num, H]
        ret = torch.zeros_like(h)
        s = 0
        for bi in range(batch_size):
            cur_len = text_len_tensor[bi].item()
            y = x[s: s + cur_len]
            z = h[s: s + cur_len]
            scores = torch.mm(z, y.t())  # [L, L]
            probs = F.softmax(scores, dim=1)
            out = z.unsqueeze(0) * probs.unsqueeze(-1)  # [1, L, H] x [L, L, 1] --> [L, L, H]
            out = torch.sum(out, dim=1)  # [L, H]
            # ret[s: s + cur_len, :] = out
            ret[s: s + cur_len, :] = z
            s += cur_len

        return ret


