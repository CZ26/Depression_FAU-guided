import argparse

from tqdm import tqdm
import pickle

import mdg

log = mdg.utils.get_logger()


def split():
    mdg.utils.set_seed(args.seed)

    sample_ids, sample_speakers, labels,\
    CNN_resnet, _, audio, au_spec, trainids, \
    testids, devids = pickle.load(open('../AVEC_features_AAAI_new.pkl', 'rb'))

    train, dev, test = [], [], []

    for vid in tqdm(trainids, desc="train"):
        train.append(mdg.Sample(vid, sample_speakers[vid], labels[vid],
                                 CNN_resnet[vid], audio[vid], au_spec[vid]))
    for vid in tqdm(devids, desc="dev"):
        dev.append(mdg.Sample(vid, sample_speakers[vid], labels[vid],
                               CNN_resnet[vid], audio[vid], au_spec[vid]))
    for vid in tqdm(testids, desc="test"):
        test.append(mdg.Sample(vid, sample_speakers[vid], labels[vid],
                               CNN_resnet[vid], audio[vid], au_spec[vid]))

    log.info("train vids:")
    log.info(sorted(trainids))
    log.info("dev vids:")
    log.info(sorted(devids))
    log.info("test vids:")
    log.info(sorted(testids))

    return train, dev, test


def main(args):
    train, dev, test = split()
    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))
    data = {"train": train, "dev": dev, "test": test}
    mdg.utils.save_pkl(data, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["avec"],
                        help="Dataset name.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    args = parser.parse_args()

    main(args)
