import numpy as np
from collections import namedtuple
import sys, traceback

import pickle
import operator

from sklearn import mixture

class BayesClassifier(object):
    
    def __init__(self):
        
        """ Initialize classifier with training data. """
        
        stateStruct = namedtuple('stateStruct', ['labels','mean', 'var','n']);
        
        self.states = stateStruct({},[],[],[]);
        
        #self.labels = []    # class labels
        #self.mean = []      # class mean
        #self.var = []       # class variances
        #self.n = 0          # nbr of classes
        
    def saveState(self):
        fn = 'labels.trn';
        with open(fn, 'wb') as ff:
            pickle.dump(self.states.labels,ff);
        ff.close();
        
        fn = 'mean.trn';
        with open(fn, 'wb') as ff:
            pickle.dump(self.states.mean,ff);
        ff.close();
        
        fn = 'var.trn';
        with open(fn, 'wb') as ff:
            pickle.dump(self.states.var,ff);
        ff.close();
        
        fn = 'n.trn';
        with open(fn, 'wb') as ff:
            pickle.dump(self.states.n,ff);
        ff.close();        
        
    def loadState(self):
        fn = 'labels.trn';
        with open(fn, 'rb') as ff:
            labs = pickle.load(ff);
        ff.close();
        
        labels = sorted(labs.items(), key=operator.itemgetter(0));
        
        fn = 'mean.trn';
        with open(fn, 'rb') as ff:
            mean = pickle.load(ff);
        ff.close();
        
        fn = 'var.trn';
        with open(fn, 'rb') as ff:
            var = pickle.load(ff);
        ff.close();
        
        fn = 'n.trn';
        with open(fn, 'rb') as ff:
            n = pickle.load(ff);
        ff.close();
        
        for ch_idx in labels:
            ch  = ch_idx[0];
            idx = ch_idx[1]; # New class index
            if ch in self.states.labels:
                print('ERROR: Entry to be loaded already exists\n');
            else:
                sIdx = len(self.states.labels);
                self.states.labels[ch] = sIdx;
                self.states.mean.append(mean[idx]);
                self.states.var.append(var[idx]);
                self.states.n.append(n[idx]);
                        
    def train(self,labels,muB,cMat_B,nSamp):
        """ Train on data (list of arrays n*dim). 
            Labels are optional, default is 0...n-1. """
        try:
            if(len(labels)):
                for ch_idx in labels:
                    ch  = ch_idx[0];
                    idx = ch_idx[1]; # New class index
                    if ch in self.states.labels:
                        sIdx   = self.states.labels[ch]; # State class index
                        cMat_A = self.states.var[sIdx];
                        muA    =  self.states.mean[sIdx];
                        coVar  = (self.states.n[sIdx]-1)*cMat_A + \
                        (nSamp[idx] -1)*cMat_B[idx] + \
                        (self.states.n[sIdx]*nSamp[idx])/(self.states.n[sIdx]+nSamp[idx]) * \
                        np.dot((muA - muB[idx]).T,(muA - muB[idx]));
                        
                        self.states.mean[sIdx] = (self.states.n[sIdx]*muA +\
                        nSamp[idx]*muB[idx])/(self.states.n[sIdx]+nSamp[idx]);
                        self.states.n[sIdx] += nSamp[idx];
                        self.states.var[sIdx] = coVar/(self.states.n[sIdx]-1);
                    else:
                        sIdx = len(self.states.labels);
                        self.states.labels[ch] = sIdx;
                        self.states.mean.append(muB[idx]);
                        self.states.var.append(cMat_B[idx]);
                        self.states.n.append(nSamp[idx]);
            else:
                raise NameError('ERROR_SIZE');
        
        except NameError:
            print('BayesClassifier.train :: Array size mismatch.\n');
            traceback.print_exc(file=sys.stdout);
            sys.exit(0);
        
    def classify(self,points):
        """ Classify the points by computing probabilities 
            for each class and return most probable label. """
        try:
            if(len(self.states.mean)==len(self.states.var)):
                mu = np.array(self.states.mean);
                cv = np.array(self.states.var);
                
                logProb = mixture.log_multivariate_normal_density(points, mu, cv, 'full');
                
                ndx = logProb.argmax(axis=1);
        
                est_labels = [];
                for n in ndx:
                    for k,v in self.states.labels.items():
                        if v==n:
                            est_labels.append(k);
                            
                #print 'est prob',est_prob.shape,self.states.labels;
                # get index of highest probability, this gives class label
                
                return est_labels, logProb;
            else:
                raise NameError('ERROR_SIZE');
        
        except NameError:
            print('BayesClassifier.train :: Array size mismatch.\n');
            traceback.print_exc(file=sys.stdout);
            sys.exit(0);
                
        


def gauss(m,v,x):
    """ Evaluate Gaussian in d-dimensions with independent 
        mean m and variance v at the points in (the rows of) x. 
        http://en.wikipedia.org/wiki/Multivariate_normal_distribution """
    
    if len(x.shape)==1:
        n,d = 1,x.shape[0];
    else:
        n,d = x.shape;
            
    # covariance matrix, subtract mean
    S = np.diag(1/v);
    x = x-m;
    # product of probabilities
    #y = np.exp(-0.5*np.diag(np.dot(x,np.dot(S,x.T))));
    Sx = np.dot(S,x.T);
    y = [];
    for i in range(n):
        ey = np.exp(-0.5*np.dot(x[i,:], Sx[:,i]));
        y.append(ey);
    
    y = np.array(y);
    # normalize and return
    return y * (2*np.pi)**(-d/2.0) / ( np.prod(np.sqrt(v)) + np.finfo(np.double).tiny);


