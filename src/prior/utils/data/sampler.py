import random
from torch.utils.data import Sampler


class RandomFoldedLengthSampler(Sampler):
    def __init__(
        self, 
        length_list,
        batch_bins,
        num_folds,
        num_replicas=1,
        rank=0,
        seed=42,
    ):
        self.length_list = length_list
        self.batch_bins = batch_bins
        self.num_folds = num_folds
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.batch_indices = []
        self.create_batch_indices()

    def create_batch_indices(self):
        random.seed(self.seed + self.epoch)
        shuffled_indices = list(range(len(self.length_list)))
        random.shuffle(shuffled_indices)
        folds = []
        for i in range(self.num_folds):
            fold_indices = shuffled_indices[i::self.num_folds]
            fold_indices = sorted(fold_indices, key=lambda idx: self.length_list[idx], reverse=True)
            folds.append(fold_indices)

        batch_indices = []
        rest_indices = []
        for fold in folds:
            seek = 0
            while True:
                first_sample = fold[seek]
                batch_size = self.batch_bins // self.length_list[first_sample]
                if seek + batch_size > len(fold):
                    rest_indices.extend(fold[seek:])
                    break
                batch_indices.append(fold[seek:seek + batch_size])
                seek += batch_size
        
        seek = 0
        while seek < len(rest_indices):
            first_sample = rest_indices[seek]
            batch_size = self.batch_bins // self.length_list[first_sample]
            batch_indices.append(rest_indices[seek:min(seek + batch_size, len(rest_indices))])
            seek += batch_size

        n = len(batch_indices)
        lack = (self.num_replicas - n % self.num_replicas) % self.num_replicas
        batch_indices.extend(random.choices(batch_indices, k=lack))
        random.shuffle(batch_indices)
        self.batch_indices = batch_indices[self.rank::self.num_replicas]

    def __iter__(self):
        self.create_batch_indices()
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
