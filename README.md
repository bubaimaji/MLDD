## Joint Representation Learning of Speech Foundation Models and Acoustic Features for Multilingual Depression Detection ##


------
###Repository Structure
notebooks contains a full demo of our method being applied to the RAVDESS dataset and a notebook to get all the plots
modules contains code for feature extraction, preprocessing, emotion classification, feature importance and regression

###Installation
To get started, follow the steps below:

Create a Conda environment with Python Python 3.10:

```bash
conda create -n myenv python=3.10
conda activate myenv

Install dependencies:

After activating the environment, run the following command to install the required dependencies:

```bash
pip install -r requirements.txt


---
Datasets:
----

The Indic-Bengali is a dataset consist of audio files of 70 subjects (30 depressed and 40 healthy individuals).

How to download
The Indic-Bengali can be downloaded at https://drive.google.com/drive/folders/1dJyKYYkIKXq8QEx9H11LRlIBMDQsxZVQ

How to use

Dataset used 5-fold speaker independent.

Each fold contains unique speaker data.

bangla_5fold_metadata.csv file contains meta data

------
DAIC-WOZ and EDAIC
------
[DAIC-WOZ and EDAIC](https://dcapswoz.ict.usc.edu/) dataset. 

------
Androids-Corpus (Italian)
------
[Androids-Corpus](https://github.com/androidscorpus/data) dataset. 



