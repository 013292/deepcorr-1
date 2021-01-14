import numpy as np
from tqdm.notebook import tqdm
import pdb

def accuracy(logits, y_truth):
    '''
    logits.shape == y_truth.shape == [batch_size,]
    logits: output from sigmoid.
    '''
    batch_size = y_truth.shape[0]
    return np.sum((logits >= 0.5) == y_truth)/batch_size

class GAN_Simulator():
    '''
    Simulate GAN on the ingress of Tor only.
    '''
    def __init__(self, flow_size=300, duplex=False):
        '''
        duplex: If True, modify all the traffic, otherwise, only change the output from here.
        '''
        self.flow_size = flow_size
        self.duplex = duplex
        return
    
    def __call__(self, dataset):
#         pdb.set_trace()
        for i in tqdm(range(len(dataset)), desc='Through GAN simulator'):
            noise_time_here = np.random.rand(self.flow_size)
            noise_size_here = np.random.rand(self.flow_size)*1000
            if self.duplex:
                noise_time_there = np.random.rand(self.flow_size)
                noise_size_there = np.random.rand(self.flow_size)*1000
            
            dataset[i]['here'][0]['->'][:self.flow_size] = (dataset[i]['here'][0]['->'][:self.flow_size] \
                                                            + noise_time_here).tolist()
            dataset[i]['there'][0]['<-'][:self.flow_size] = (dataset[i]['there'][0]['<-'][:self.flow_size] \
                                                            + noise_time_here).tolist()
            dataset[i]['here'][1]['->'][:self.flow_size] = (dataset[i]['here'][1]['->'][:self.flow_size] \
                                                            + noise_size_here).tolist()
            dataset[i]['there'][1]['<-'][:self.flow_size] = (dataset[i]['there'][1]['<-'][:self.flow_size] \
                                                            + noise_size_here).tolist()
            if self.duplex:
                dataset[i]['there'][0]['->'][:self.flow_size] = (dataset[i]['there'][0]['->'][:self.flow_size] \
                                                                + noise_time_there).tolist()
                dataset[i]['here'][0]['<-'][:self.flow_size] = (dataset[i]['here'][0]['<-'][:self.flow_size] \
                                                                + noise_time_there).tolist()
                dataset[i]['there'][1]['->'][:self.flow_size] = (dataset[i]['there'][1]['->'][:self.flow_size] \
                                                                + noise_size_there).tolist()
                dataset[i]['here'][1]['<-'][:self.flow_size] = (dataset[i]['here'][1]['<-'][:self.flow_size] \
                                                                + noise_size_there).tolist()
        return dataset