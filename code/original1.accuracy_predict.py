import numpy as np
from accuracy_predict_resources import retrieval

def predict(dataset_feature, model_feature):
    accuracy_feature = np.concatenate((dataset_feature, model_feature))
    r = retrieval.retrievalEngine()
    r.setClassifier('RBFSVR')
    accuracy = r.predict(accuracy_feature.reshape(1,-1))
    return accuracy

'''
if __name__ == '__main__':
    import json
    nlds_file_path = '../data/nlds/1aadb18e2fde21d34c0ec868315dd824/1aadb18e2fde21d34c0ec868315dd824.json'
    with open(nlds_file_path) as dump:
        model_nlds = json.load(dump)
    dataset_name = '../data/fmnist_test'
    predict(dataset_name, model_nlds)
'''