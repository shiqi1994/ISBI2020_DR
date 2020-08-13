# ISBI2020 Challenge: Diabetic Retinopathy Assessment Grading and Diagnosis

This solution is mainly in 3 stages:
1. Ensemble EfficientNet-b3, EfficientNet-b4, EfficientNet-b5 and pretrain on Kaggle2015 DR dataset, finetune on ISBI2020 DR dataset. 
2. Train XGBoost, SVR, CATBoost, LightGBM models on logits output of ensembled models. 
3. Voting on results from stage 2.

## Getting Started

This implementation is under the enviroment of ```python3.6.8``` and ```pytorch 1.0.1.post2```. 
First, install from source, and make sure you are in this dictionary:
```
git clone https://github.com/shiqi1994/ISBI2020_DR.git
```
### Prerequisites

Here are some packages you need to install.
```
pip install -r requirements.txt
```

### Data Preprocessing
Download the [ISBI2020 DR task1 dataset](https://isbi.deepdr.org/data.html). And unzip all images into one folder, then do the [Ben-Graham's-preprocessing](https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy#2.-Try-Ben-Graham's-preprocessing-method.) to both training and validation dataset. 
Open ```./toolkit/preprocessing.py``` and change the 'dictionary', 'image size' and 'sigmaX' inside this script, then excute
```
python ./toolkit/preprocessing.py
```
Next, we rearrange the images according to the given labels in this way, which is the generic data arrangment for ```torchvision.dataset.DatasetFolder```. Note that the folder name of ```Train``` and ```Test``` may be different. Please see ```data_util.py``` for more datails.

Open ```isbi2020_data_rearrangement.py```  and change the ‘dictionary’ inside, then excute
```
python ./toolkit/isbi2020_data_rearrangement.py
```
Then the data should be rearranged as following:
```
├── Data
    ├── Train
        ├── class_x
            ├── xxx.jpg
            ...
        ├── class_y
            ├── xxxjpg
            ...
    ├── Test
        ├── class_x
            ├── xxx.jpg
            ...
        ├── class_y
            ├── xxx.jpg
            ...
```
## Stage1: Training and Testing
The following proceess is to be used on efficentnet-b3, efficientnet-b4, efficientnet-b5. Modify the model in ```'train_config.py'``` and ```'test_config.py'```.

Since either train or test phrase, mean, std, eigvalue and eigvector of dataset is necessary for data augmentation. Use ```./toolkit/meanstd.py``` and ```./toolkit/eigval_vec.py``` to compute that. Then set the results in ```data_utils.py```.

### Train with cross validation

Set ```'CROSS_VALIDATION'=True``` and assign ```'CV_SEED'``` in ```train_config.py```, then excute

```
python main.py --MODE=TRAIN
```

### Train without cross validation

Set ```'CROSS_VALIDATION'=False``` and in ```train_config.py```. 
And make sure that you have set the train and test folder correctly.
```
python main.py --MODE=TRAIN
```
### Test
Set ```'PRETRAINED_PATH'``` , the directory of saved model, then excute
```
python main.py --MODE=TEST
```
## Stage2&3: Boosting and Voting
Make directory for training and testing data for stage2 as following:
```
mkdir Boost_Data/Train
mkdir Boost_Data/Test
```
Copy the csv file which contain the output of stage1 to ```./Boost_Data/Train/```.
Here is an example:
```
cd Results/2020_02_19_14_18_14
cp efficientnet_b*_fold*.csv ~/Doucuments/ISBI2020/Boost_Data/Train
```
Prepare the data for validation on boosting algorithm. ```val_config.py``` can be modified.
```
python main.py --MODE='VAL'
```
Copy the csv file which contain the validation results to ```./Boost_Data/Val/```. 

Then excute different boosting algorithms
```
python boost.py
```
Boosting parameters can be modified inside ```boost.py```. Then a csv file containing final result is created under the directory ```./fianl_result.csv```.





