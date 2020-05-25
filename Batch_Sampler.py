import numpy as np
import torch
import pdb

class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, dataset, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.dataset = dataset
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        #pdb.set_trace()

        # create a vector numel_per_class and fill it with the number of samples each class has
        self.numel_per_class = np.zeros(len(self.dataset.classes)+1, dtype=int)
        self.numel_per_class[0] = 0
        self.numel_per_class[len(self.dataset.classes)] = len(dataset)
        i = 1
        for idx in range(len(dataset)-1):
            if (dataset[idx][1] != dataset[idx+1][1]):
                self.numel_per_class[i] = idx+1
                i = i+1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for _ in range(self.iterations):
            batch = torch.LongTensor(cpi*spc)
            c_idxs = torch.randperm(len(self.dataset.classes))[:cpi]
            for i, c in enumerate(c_idxs):
                s = slice(i * spc, (i + 1) * spc)
                batch[s] = self.numel_per_class[c] + torch.randperm(self.numel_per_class[c+1]-self.numel_per_class[c])[:spc]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations