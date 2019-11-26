import math
import numpy as np
from skimage.feature import multiblock_lbp as mlbp
from skimage.feature import daisy
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import gist #From the link: https://github.com/tuttieee/lear-gist-python
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.externals import joblib
import config

#model_root_path = './services/models/'
model_root_path = config.models_path

class datasetFeature():
    """Computes and stores the feature values"""
    def __init__(self, opt, dataloader):
        self.opt = opt
        self.dataloader = dataloader

    def extractFeatures(self, dsetname):
        self.getNonDeepLearningFeatures(self.dataloader, dsetname)

    def extractFeaturesEverything(self):
        mean, std = self.get_mean_and_std(self.dataloader)
        lom = self.getMoments(self.dataloader, mean, std)
        self.getNonDeepLearningFeatures(self.dataloader)

        #print(lom)
        lomnp = lom[0]
        for l in lom[1:]:
            lomnp = np.hstack((lomnp,l))
        #print(lomnp)
        return mean.numpy(), std.numpy(), lomnp

    def get_mean_and_std(self, dataloader):
        '''Compute the mean and std value of dataset.'''
        mean = torch.zeros(3)
        std = torch.zeros(3)
        len_dataset = 0

        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            len_dataset += 1
            for i in range(inputs.size(1)):
                mean[i] += inputs[:,i,:,:].mean()
                std[i] += inputs[:,i,:,:].std()

            if len_dataset >= self.opt['sampleSize']:
                break

        mean.div_(len_dataset)
        std.div_(len_dataset)
        #print(mean)
        return mean, std

    def getMoments(self, dataloader, mean, std):
        lom = []

        for i in range(self.opt['order']-1):
            lom.append(torch.zeros(3))

        len_dataset = 0

        print('==> Computing moments...')
        for inputs, targets in dataloader:
            len_dataset += 1
            for j in range(len(lom)):
                for i in range(inputs.size(1)):
                    val = torch.add(inputs[:,i,:,:],-1*mean[i])
                    (lom[j])[i] += torch.pow(val, j+1).mean()

            if len_dataset >= self.opt['sampleSize']:
                break

        for j in range(len(lom)):
            lom[j].div_(len_dataset)
            lom[j].div_(torch.pow(std,j+1))

        for j in range(len(lom)):
            for i in range(3):
                if math.isnan(lom[j][i]):
                    lom[j][i] = 0

        return lom

    def getNonDeepLearningFeatures(self, dataloader, dsetname):
        daisyfeat = []
        lbpfeat = []
        gistfeat = []
        orbfeat = []
        hogfeat = []
        Y = []
        len_dataset = 0
        #descriptor_extractor = ORB(n_keypoints=self.opt['nKeypoints'])

        print('==> Computing image features for', dsetname,' ...')
        for inputs, targets in dataloader:
            for i in range(inputs.size(0)):
                len_dataset += 1
                img = inputs[i,:,:,:].transpose(0,2).numpy()
                img2 = img[:,:,0]
                f1, _ = daisy(img2, step=int(img.shape[0]/2), radius=self.opt['daisyRadius'], rings=self.opt['daisyRings'], histograms=self.opt['daisyHistograms'],
                         orientations=self.opt['daisyOrientations'], visualize=False)
                daisyfeat.append(f1.flatten())
                #lbpfeat.append(mlbp(img2,0,0,self.opt['lbpWidth'],self.opt['lbpHeight']))
                lbp = local_binary_pattern(img2, self.opt['lbpP'], self.opt['lbpR'], self.opt['lbpMethod'])
                lbp_hist, _ = np.histogram(lbp, normed=True, bins=self.opt['lbpP'] + 2, range=(0, self.opt['lbpP'] + 2))
                lbpfeat.append(lbp_hist)
                gistfeat.append(gist.extract((img*255).astype('uint8')))
                hogfeat.append(hog(img2, orientations=self.opt['hogOrientations'], pixels_per_cell=(self.opt['hogCellSize'], self.opt['hogCellSize']), cells_per_block=(self.opt['hogNumCells'], self.opt['hogNumCells']), block_norm=self.opt['hogBlockNorm'], feature_vector=True))
                Y.append(targets[i].numpy())
                
                #descriptor_extractor.detect_and_extract((img2*255).astype('uint8'))
                #orbfeat.append(descriptor_extractor.descriptors.flatten())

            if len_dataset >= self.opt['sampleSize']:
                break
        self.daisyfeat = np.array(daisyfeat)
        self.lbpfeat = np.array(lbpfeat)
        self.gistfeat = np.array(gistfeat)
        self.hogfeat = np.array(hogfeat)
        self.Y = np.array(Y)
        print("Daisy Features: ", self.daisyfeat.shape)
        print("LBP Features: ", self.lbpfeat.shape)
        print("GIST Features: ", self.gistfeat.shape)
        print("HOG Features: ", self.hogfeat.shape)
        print("Target labels: ", self.Y.shape)
       
        return


class retrievalEngine():
    """Computes and stores the classifier values"""
    def __init__(self, opt):
        self.opt = opt
        self.numDsets = 0
        self.datasetLabel = 0
        self.labelDict = {}
        self.revlabelDict = {}

    def addTrain(self, X, Y, datasetName, featureType):
        self.featureType = featureType
        if datasetName not in self.labelDict.keys():
            self.datasetLabel += 1
            print(self.datasetLabel)
            print(self.datasetLabel)
            self.labelDict[datasetName] = self.datasetLabel
            self.revlabelDict[self.datasetLabel] = datasetName

        try:
            self.X = np.concatenate((self.X, X))
            self.YL = np.concatenate((self.YL, Y))
            YD = np.ones(Y.shape[0])*self.datasetLabel
            self.YD = np.concatenate((self.YD, YD))
            #print(self.X.shape)
            #print(self.YL.shape)
            #print(self.YD.shape)
            assert(self.X.shape[0] == self.YL.shape[0])
            assert(self.X.shape[0] == self.YD.shape[0])

        except AttributeError:
            self.X = X
            self.YL = Y
            self.YD = np.ones(Y.shape[0])*self.labelDict[datasetName]
            assert(self.X.shape[0] == self.YL.shape[0])
            assert(self.X.shape[0] == self.YD.shape[0])
            #print("I was here!")

        #print(self.labelDict)
        return

    def addTest(self, X, Y, datasetName, featureType):
        self.featureType = featureType
        if datasetName not in self.labelDict.keys():
            self.datasetLabel += 1
            self.labelDict[datasetName] = self.datasetLabel
        
        try:
            self.XVal = np.concatenate((self.XVal, X))
            self.YLVal = np.concatenate((self.YLVal, Y))
            YD = np.ones(Y.shape[0])*self.labelDict[datasetName]
            self.YDVal = np.concatenate((self.YDVal, YD))
            #print(self.XVal.shape)
            #print(self.YDVal.shape)
            #print(set(self.YDVal))
            #print(self.YDVal)
            assert(self.XVal.shape[0] == self.YLVal.shape[0])
            assert(self.XVal.shape[0] == self.YDVal.shape[0])
        except AttributeError:
            self.XVal = X
            self.YLVal = Y
            self.YDVal = np.ones(Y.shape[0])*self.labelDict[datasetName]
            assert(self.XVal.shape[0] == self.YLVal.shape[0])
            assert(self.XVal.shape[0] == self.YDVal.shape[0])
            
        return

    def train(self):
        self.norm = Normalizer()
        self.XT = self.norm.fit_transform(self.X)
        print("=> Training model " + self.featureType + '_' + self.classificationType + '.pkl' + " ...")
        print(set(self.YD))
        print(set(self.YL))
        self.model.fit(self.XT, self.YD)
        joblib.dump(self.norm, model_root_path + self.featureType + '_' + self.classificationType + '_norm.pkl') 
        joblib.dump(self.model, model_root_path + self.featureType + '_' + self.classificationType + '.pkl') 
        return

    def test(self):
        self.norm = joblib.load(model_root_path + self.featureType + '_' + self.classificationType + '_norm.pkl')
        self.model = joblib.load(model_root_path + self.featureType + '_' + self.classificationType + '.pkl')
        self.norm.transform(self.XVal)
        print("=> Testing model " + self.featureType + '_' + self.classificationType + '.pkl' + " ...")
        YPred = self.model.predict(self.XVal)
        #print('=> Getting confusion matrix...')
        self.cf = confusion_matrix(self.YDVal,YPred)
        #print(self.cf)
        self.acc = accuracy_score(YPred,self.YDVal)
        #print("=> Accuracy: ",self.acc)
        self.f1s = f1_score(YPred,self.YDVal, average='micro')
        #print("=> F1-score: ",self.f1s)

        return YPred

    def classification(self, classificationType):
        self.classificationType = classificationType
        if classificationType == 'LinearSVM':
            model = LinearSVC(random_state=0)

        elif classificationType == 'Adaboost':
            model = AdaBoostClassifier(n_estimators=100)
    
        elif classificationType == 'MultinomialNB':
            model = MultinomialNB()

        elif classificationType == 'RandomForest':
            model = RandomForestClassifier(max_depth=30, n_estimators=150, n_jobs=-1)

        elif classificationType == 'GradientBoost':
            model = GradientBoostingClassifier(max_depth=30, n_estimators=150)

        self.model = model
