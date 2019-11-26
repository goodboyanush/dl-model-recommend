# "You might also like this model": Data Driven Approach for Recommending Deep Learning Models for Unknown Datasets

This repo makes available the original implementation of the paper - `"You might also like this model": Data Driven Approach for Recommending Deep Learning Models for Unknown Datasets`

## Install Prerequisites

```
> git clone git@github.ibm.com:aimodels/catalog-ms-recommendation-paper.git

> cd catalog-ms-recommendation-paper

> pip install -r requirements.txt
```

## Quick Running

```
> cd code/

> python model_recommend.py
```

- The input dataset is `FMNIST` and placed here: `./data/fmnist_test.zip`

- The models for which accuracy are to be predicted are kept here: `./data/nlds/` (There are 5 files)

- The output of the code will be like this:

```json
==> Computing image features for fmnist_test.zip  ...
Daisy Features:  (256, 112)
LBP Features:  (256, 18)
GIST Features:  (256, 960)
HOG Features:  (256, 324)
Target labels:  (256,)
=> Testing model lbp_LinearSVM.pkl ...
=> Testing model daisy_LinearSVM.pkl ...
=> Testing model gist_LinearSVM.pkl ...
=> Testing model lbp_RandomForest.pkl ...
=> Testing model daisy_RandomForest.pkl ...
=> Testing model gist_RandomForest.pkl ...
=> Testing model lbp_MultinomialNB.pkl ...
=> Testing model daisy_MultinomialNB.pkl ...
=> Testing model gist_MultinomialNB.pkl ...
=> Testing model lbp_Adaboost.pkl ...
=> Testing model daisy_Adaboost.pkl ...
=> Testing model gist_Adaboost.pkl ...
=> Testing model lbp_GradientBoost.pkl ...
=> Testing model daisy_GradientBoost.pkl ...
=> Testing model gist_GradientBoost.pkl ...
model recommend: dataset features extracted
Predicting model accuracy ... 
=> Testing model...   cifar10_68
model recommend: model features extracted -  (2048,)
model recommend: accuracy predicted -  [0.26946919]
=> Testing model...   cifar10_908
model recommend: model features extracted -  (2048,)
model recommend: accuracy predicted -  [0.27060888]
=> Testing model...   cifar10_1
model recommend: model features extracted -  (2048,)
model recommend: accuracy predicted -  [0.27076088]
=> Testing model...   cifar10_801
model recommend: model features extracted -  (2048,)
model recommend: accuracy predicted -  [0.27229994]
=> Testing model...   cifar10_933
model recommend: model features extracted -  (2048,)
model recommend: accuracy predicted -  [0.26962945]
Sorting for the best models ... 
[{"_id": "cifar10_801", "accuracy": 27.2299940725517}, {"_id": "cifar10_1", "accuracy": 27.076087560274477}, {"_id": "cifar10_908", "accuracy": 27.060888232986326}, {"_id": "cifar10_933", "accuracy": 26.962945414430774}, {"_id": "cifar10_68", "accuracy": 26.946919074586763}]
```
