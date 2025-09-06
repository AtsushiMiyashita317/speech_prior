import random
from torch.utils.data import Sampler


class FoldedLengthBatchSampler(Sampler):
    def __init__(
        self, 
        length_list,
        batch_bins,
        num_folds,
        num_outliers=0,
        shuffle=True,
        num_replicas=1,
        rank=0,
        seed=42,
    ):
        self.length_list = length_list
        self.batch_bins = batch_bins
        self.num_folds = num_folds
        self.num_outliers = num_outliers
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.sorted_indices = sorted(range(len(length_list)), key=lambda idx: length_list[idx], reverse=True)
        self.batch_indices = []
        self.create_batch_indices()
        
    def generate_batch_indices(self, indices, num_folds):
        folds = []
        for i in range(self.num_folds):
            fold_indices = indices[i::self.num_folds]
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
                if seek >= len(fold):
                    break
                
        return batch_indices, rest_indices

    def create_batch_indices(self):
        random.seed(self.seed + self.epoch)
        if self.num_outliers > 0:
            outliers = self.sorted_indices[:self.num_outliers]
            indices = self.sorted_indices[self.num_outliers:]
        else:
            indices = self.sorted_indices[:]
            outliers = []
            
        if self.shuffle:
            random.shuffle(indices)
            random.shuffle(outliers)
        
        batch_indices, rest_indices = self.generate_batch_indices(indices, self.num_folds)        
        rest_batch_indices, rest_indices = self.generate_batch_indices(rest_indices, (self.num_folds * len(rest_indices)) // len(indices) + 1)
        batch_indices.extend(rest_batch_indices)
        rest_indices = sorted(rest_indices, key=lambda idx: self.length_list[idx], reverse=True)
        
        seek = 0
        while seek < len(rest_indices):
            first_sample = rest_indices[seek]
            batch_size = self.batch_bins // self.length_list[first_sample]
            batch_indices.append(rest_indices[seek:min(seek + batch_size, len(rest_indices))])
            seek += batch_size
            
        seek = 0
        while seek < len(outliers):
            first_sample = outliers[seek]
            batch_size = self.batch_bins // self.length_list[first_sample]
            batch_indices.append(outliers[seek:min(seek + batch_size, len(outliers))])
            seek += batch_size

        n = len(batch_indices)
        lack = (self.num_replicas - n % self.num_replicas) % self.num_replicas
        if self.shuffle:
            batch_indices.extend(random.choices(batch_indices, k=lack))
            random.shuffle(batch_indices)
        else:
            batch_indices.extend(batch_indices[:lack])
        self.batch_indices = batch_indices[self.rank::self.num_replicas]

    def __iter__(self):
        self.create_batch_indices()
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
