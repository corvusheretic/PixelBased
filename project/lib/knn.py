import numpy as np

class KnnClassifier(object):
    
    def __init__(self,labels,samples):
        """ Initialize classifier with training data. """
        
        self.labels = labels
        self.samples = samples
    
    def classify(self,point,k=3):
        """ Classify a point against k nearest 
            in the training data, return label. """
        
        # compute distance to all training points
        dist = np.array([L2dist(point,s) for s in self.samples])
        
        # sort them
        ndx = dist.argsort()
        
        # use dictionary to store the k nearest
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label,0)
            votes[label] += 1
        
        labList=[]; votList=[];
        for key, value in votes.iteritems():
            labList.append(key);
            votList.append(value);
        
        return labList[votList.index(np.max(votList))];


def L2dist(p1,p2):
    return np.sqrt( np.sum( (p1-p2)**2) )

def L1dist(v1,v2):
    return np.sum(np.abs(v1-v2))