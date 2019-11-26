import joblib
import sklearn
import pickle
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier,MLPRegressor
from scipy.sparse import vstack
import scipy.sparse as sps
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Ridge,LinearRegression
import config

#model_root_path = './services/models/'
model_root_path = config.models_path

class retrievalEngine():
    """Retrieval"""
    def __init__(self):
        self.numDsets = 0
        self.datasetLabel = -1
        self.labelDict = {}
        self.revlabelDict = {}
        self.YFDict = {}
        self.revYFDict = {}
        self.numFineClasses = -1
        self.norm = Normalizer()

    def addData(self, X, Y, datasetName, split):
        if datasetName not in self.labelDict.keys():
            self.datasetLabel += 1
            self.labelDict[datasetName] = int(self.datasetLabel)
            self.revlabelDict[self.datasetLabel] = datasetName

        YD = np.ones(Y.shape[0])*self.labelDict[datasetName]
        YF = np.zeros(Y.shape[0])

        for i in range(YD.shape[0]):
            if (int(YD[i]),int(Y[i])) not in self.YFDict.keys():
                self.numFineClasses += 1
                self.YFDict[(int(YD[i]),int(Y[i]))] = int(self.numFineClasses)
                self.revYFDict[self.numFineClasses] = (int(YD[i]),int(Y[i]))

            YF[i] = self.YFDict[(int(YD[i]),int(Y[i]))]

        try: 
            if split == 'train':
                if sps.issparse(X):
                    self.X = vstack((self.X,X))
                else:
                    self.X = np.vstack((self.X, X))
                self.YL = np.hstack((self.YL, Y))
                self.YD = np.hstack((self.YD, YD))
                self.YF = np.hstack((self.YF, YF))
                assert(self.X.shape[0] == self.YL.shape[0])
                assert(self.X.shape[0] == self.YD.shape[0])
                assert(self.X.shape[0] == self.YF.shape[0])
            else:
                if sps.issparse(X):
                    self.XVal = vstack((self.XVal,X))
                else:
                    self.XVal = np.vstack((self.XVal, X))
                self.YLVal = np.hstack((self.YLVal, Y))
                self.YDVal = np.hstack((self.YDVal, YD))
                self.YFVal = np.hstack((self.YFVal, YF))
                assert(self.XVal.shape[0] == self.YLVal.shape[0])
                assert(self.XVal.shape[0] == self.YDVal.shape[0]) 
                assert(self.XVal.shape[0] == self.YFVal.shape[0])   

        except AttributeError:
            if split == 'train':
                self.X = X
                self.YL = Y
                self.YD = YD
                self.YF = YF
                assert(self.X.shape[0] == self.YL.shape[0])
                assert(self.X.shape[0] == self.YD.shape[0])
                assert(self.X.shape[0] == self.YF.shape[0])
            else:
                self.XVal = X
                self.YLVal = Y
                self.YDVal = YD
                self.YFVal = YF
                assert(self.XVal.shape[0] == self.YLVal.shape[0])
                assert(self.XVal.shape[0] == self.YDVal.shape[0])
                assert(self.XVal.shape[0] == self.YFVal.shape[0])

    def train(self, X, Y):
        XT = self.norm.fit_transform(X)
        #XT = X
        print("=> Training model...")
        self.model.fit(XT, Y)

    def predict(self, X):
        self.model = joblib.load(model_root_path + 'RBFSVR.pkl')
        XTVal = self.norm.transform(X)
        #XTVal = X
        self.YPred = self.model.predict(XTVal)
        return self.YPred

    def evaluate(self, YPred, YTrue, mode='all'):
        if mode == 'mae' or mode =='all':
            self.mae = mean_absolute_error(YPred, YTrue)
            #print(self.cf)
        if mode == 'mse' or mode == 'all':
        	self.mse = mean_squared_error(YPred, YTrue)

        if mode == 'r2' or mode =='all':
            self.r2 = r2_score(YPred, YTrue)

        if mode == 'evs' or mode=='all':
           self.evs = explained_variance_score(YPred, YTrue)

    def setClassifier(self, classifierName):
        self.classifierName = classifierName
        if classifierName == 'RBFSVR':
            model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        
        elif classifierName == 'LinearSVR':
            model = LinearSVR(C=1e3)
		
        elif classifierName == 'PolySVR':
            model = SVR(kernel='poly', C=1e3, degree=3)

        elif classifierName == 'MLP':
            model = MLPRegressor( alpha=1e-5, hidden_layer_sizes=(256, 128), random_state=1)
    
        elif classifierName == 'RidgeRegress':
            model = Ridge(alpha=0.5)

        elif classifierName == 'RandomForest':
            model = RandomForestRegressor(max_depth=25, n_estimators=250, n_jobs=-1)

        elif classifierName == 'GradientBoost':
            model = GradientBoostingRegressor(n_estimators=250)

        elif classifierName == 'AdaBoost':
            model = AdaBoostRegressor(n_estimators=250)

        elif classifierName == 'MeanWaalaClassifier':
            model = LinearRegression()

        self.model = model
