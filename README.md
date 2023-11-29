# Complex-NNphase

### ComplexNNphase is a python based complex-valued machine learning model for phase retreival problem. 

### Author: Xi Yu

### Emails: xyu1@bnl.gov, yuxi120407@gmail.com

### For using the ComplexNNphase model, the following software or package should be installed:

01.     Name: Python 
        Version: 3.8.0
        Home-page: https://www.python.org/
 
01.     Name: Pytorch 
        Version: 1.8.0
        Home-page: https://www.python.org/

03.     Name: scipy
	Version: 1.4.1
	Home-page: https://www.scipy.org

04.     Name: tifffile
	Version: 1.4.1
	Home-page: https://pypi.org/project/tifffile/

#### 1. "Simulated Date" folder contains the script for simulating the traning dataset for supervised learning.
#### 2. Codes of supervised learning on simluate data are in the "Supervised learning" folder.
#### 3. Codes of Unsupervised learning on real experimetnal data are in the "Unspervised learning" folder.

### The order of runing codes are as follows:
#### Step1. run the 'generate_data.py' in "Simulated Date" folder to generate simulated data.
#### Step2. Trained complex-valued NN with simulted data by runing 'train.py' in "Supervised learning" folder and saved the pre-trained C-CNN model.
#### Step3. Trained complex-valued NN with real expeimetnal data by running 'train.py' in the "Unspervised learning" folder.
