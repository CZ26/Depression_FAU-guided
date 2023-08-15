import math
import random

import torch


class Dataset:

    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {'M': 0, 'F': 1}

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        len_tensor = torch.tensor([len(s.audio) for s in samples]).long()
        mx = torch.max(len_tensor).item()

        cnnresnet_tensor = torch.zeros((batch_size, mx, 2048))
        auspec_tensor= torch.zeros((batch_size, mx, 48))
        audio_tensor = torch.zeros((batch_size, mx, 100))
        

        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        for i, s in enumerate(samples):
            cur_len = len(s.CNN_resnet)
            tmp = [torch.from_numpy(t).float() for t in s.CNN_resnet]
            tmp = torch.stack(tmp)
            cnnresnet_tensor[i, :cur_len, :] = tmp

            tmp = [torch.from_numpy(t).float() for t in s.audio]
            tmp = torch.stack(tmp)
            audio_tensor[i, :cur_len, :] = tmp

            tmp = [torch.from_numpy(t).float() for t in s.au_spec]
            tmp = torch.stack(tmp)
            auspec_tensor[i, :cur_len, :] = tmp


            speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker])
            labels.extend(s.label)

        label_tensor = torch.tensor(labels).float()
        data = {
            "len_tensor": len_tensor,
            "cnnresnet_tensor": cnnresnet_tensor,
            "audio_tensor":audio_tensor,
            "auspecspec_tensor":auspec_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor
        }

        return data

    def shuffle(self):
        random.shuffle(self.samples)




