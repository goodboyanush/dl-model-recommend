import argparse
import numpy as np
import config
import json
from os import listdir
from os.path import isfile, isdir, join
import dataset_similarity
import accuracy_predict
from model2vec import get_representation

def get_nlds_files():
    all_nlds = []
    nlds_path = config.nlds_data_path
    nldsfiles = [f for f in listdir(nlds_path) if not f.startswith('.')]
    for files in nldsfiles:
        with open(join(nlds_path,files), 'r') as f:
            jsondata = json.load(f)
        if "layers" in jsondata["nldsJson"]:
            all_nlds.append({"_id": files, "nlds": jsondata})
    return all_nlds

def log_status(message):
    print(message)
    return None

def recommend(data_path):
    opt = json.loads(json.dumps(config.model_params))
    opt['dataDir'] = data_path

    # Get dataset features
    log_status("Extracting dataset features ... ")
    dataset_feature = dataset_similarity.dataset_feature(opt)
    print("model recommend: dataset features extracted")
    
    # Get model features
    list_of_nlds = get_nlds_files()
    model_accuracies = []
    log_status("Predicting model accuracy ... ")
    for i, nlds in enumerate(list_of_nlds):
        print("=> Testing model...  ", nlds["_id"])
        #model_feature = np.random.rand(2048,)
        #with open('./services/models/modelX.pkl', 'rb') as handle:
        #    mX = pickle.load(handle)
        model_feature = get_representation(nlds["nlds"])
        print("model recommend: model features extracted - ", model_feature.shape)

        # Get accuracy
        accuracy = accuracy_predict.predict(dataset_feature, model_feature)
        print("model recommend: accuracy predicted - ", accuracy)
        model_accuracies.append({'_id': nlds["_id"].split(".")[0], 'accuracy': accuracy[0]*100})
    
    log_status("Sorting for the best models ... ")
    # Sorting according to accuracies
    model_accuracies = sorted(model_accuracies, key=lambda k: k.get('accuracy', 0), reverse=True)
    print(model_accuracies)
    return model_accuracies

if __name__ == '__main__':
    recommend("/Users/anush/Documents/projects/catalog/catalog-ms-model-recommendation/services/fmnist_test.zip")


