from dataset_similarity_resources import datasetList
from dataset_similarity_resources import retrieval
import argparse
import numpy as np
import os, shutil
clfs = ['LinearSVM','RandomForest','MultinomialNB','Adaboost','GradientBoost']
dsetnames = ['mnist', 'cifar10', 'cifar100', 'stl', 'svhn', 'fmnist']

def dataset_feature(opt):
    if(os.path.splitext(opt['dataDir'])[1] == ".zip"):
        new_dataset = datasetList.NewDatasetZip(opt)
    else:
        new_dataset = datasetList.NewDataset(opt)

    # Define feature extractor and extract features
    feature_extractor = retrieval.datasetFeature(opt,new_dataset.dataLoader)
    feature_extractor.extractFeatures(opt['dataDir'].split('/')[-1]) 

    # Define one classifier for each feature
    retr_feat1 = retrieval.retrievalEngine(opt) #LBP
    retr_feat2 = retrieval.retrievalEngine(opt) #Daisy
    retr_feat3 = retrieval.retrievalEngine(opt) #GIST

    retr_feat1.addTest(feature_extractor.lbpfeat, feature_extractor.Y, 'fmnist_test', 'lbp')
    retr_feat2.addTest(feature_extractor.daisyfeat, feature_extractor.Y, 'fmnist_test', 'daisy')
    retr_feat3.addTest(feature_extractor.gistfeat, feature_extractor.Y, 'fmnist_test', 'gist')

    pred_counts = []
    for clf in clfs:
        retr_feat1.classification(clf)
        retr_feat2.classification(clf)
        retr_feat3.classification(clf)	
        predictions1 = retr_feat1.test()
        predictions2 = retr_feat2.test()
        predictions3 = retr_feat3.test()
        pred_counts.append(sum(predictions1))
        pred_counts.append(sum(predictions2))
        pred_counts.append(sum(predictions3))
    
    pred_counts = np.array(pred_counts)
    
    shutil.rmtree(new_dataset.newdirname)
    return pred_counts

def similarity(opt):
    if(os.path.splitext(opt.dataDir)[1] == ".zip"):
        new_dataset = datasetList.NewDatasetZip(opt)
    else:
        new_dataset = datasetList.NewDataset(opt)

    # Define feature extractor and extract features
    feature_extractor = retrieval.datasetFeature(opt,new_dataset.dataLoader)
    feature_extractor.extractFeatures(opt.dataDir.split('/')[-1]) 

    # Define one classifier for each feature
    retr_feat1 = retrieval.retrievalEngine(opt) #LBP
    retr_feat2 = retrieval.retrievalEngine(opt) #Daisy
    retr_feat3 = retrieval.retrievalEngine(opt) #GIST

    retr_feat1.addTest(feature_extractor.lbpfeat, feature_extractor.Y, 'fmnist_test', 'lbp')
    retr_feat2.addTest(feature_extractor.daisyfeat, feature_extractor.Y, 'fmnist_test', 'daisy')
    retr_feat3.addTest(feature_extractor.gistfeat, feature_extractor.Y, 'fmnist_test', 'gist')

    pred_counts = np.zeros(len(dsetnames)-1)
    for clf in clfs:
        retr_feat1.classification(clf)
        retr_feat2.classification(clf)
        retr_feat3.classification(clf)	
        predictions1 = retr_feat1.test()
        predictions2 = retr_feat2.test()
        predictions3 = retr_feat3.test()

        for ii in range(len(pred_counts)):
            pred_counts[ii] = pred_counts[ii] + (predictions1 == ii+1).sum()
            pred_counts[ii] = pred_counts[ii] + (predictions2 == ii+1).sum()
            pred_counts[ii] = pred_counts[ii] + (predictions3 == ii+1).sum()


    sorted_indices = np.argsort(pred_counts)[::-1]
    output = {}
    for cnt, ii in enumerate(sorted_indices):
        output[str(cnt+1)] = {"dataset": dsetnames[ii], "confidence": pred_counts[ii]/sum(pred_counts)}

    print(output)
    return output



########### Classes for offline learning and tasks ###############
########### If you need to add additional classifiers in the ensemble, do it here ###############
def similarity_train():
    parser = myargparser()
    opt = parser.parse_args()  

    d1 = datasetList.MNIST(opt)
    d2 = datasetList.CIFAR10(opt)
    d3 = datasetList.CIFAR100(opt)
    d4 = datasetList.STL(opt)
    d5 = datasetList.SVHN(opt)

    dsets = [d1,d2,d3,d4,d5]
    splits = ['train']

    r1 = retrieval.retrievalEngine(opt)
    r2 = retrieval.retrievalEngine(opt)
    r3 = retrieval.retrievalEngine(opt)

    # Define feature extractor and extract features
    for i, dset in enumerate(dsets):
        fi = retrieval.datasetFeature(opt,dset.trainLoader)
        fi.extractFeatures(dsetnames[i])
        r1.addTrain(fi.lbpfeat,fi.Y, dsetnames[i], 'lbp') 
        r2.addTrain(fi.daisyfeat,fi.Y, dsetnames[i], 'daisy')
        r3.addTrain(fi.gistfeat, fi.Y, dsetnames[i], 'gist')

    # Define one classifier for each feature
    for clf in clfs:
        r1.classification(clf)
        r2.classification(clf)
        r3.classification(clf)	
        r1.train()
        r2.train()
        r3.train()

    # Testing part of the trained models
    d6 = datasetList.FashionMNIST(opt)
    fi_Test = retrieval.datasetFeature(opt,d6.trainLoader)
    fi_Test.extractFeatures('fmnist')
    r1.addTest(fi_Test.lbpfeat,fi.Y, 'fmnist', 'lbp') 
    r2.addTest(fi_Test.daisyfeat,fi.Y, 'fmnist', 'daisy')
    r3.addTest(fi_Test.gistfeat, fi.Y, 'fmnist', 'gist')
    for clf in clfs:
        r1.test()
        r2.test()
        r3.test()

if __name__ == '__main__':
    similarity('../data/fmnist_test')
    #similarity_train()
