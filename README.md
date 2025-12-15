# Joint Representation Learning of Speech Foundation Models and Acoustic Features for Multilingual Depression Detection

---

### Repository Structure
 
  This repository contains full codes of our method for Depression Detection along with generating all plots.

- `Bengali Folder`  
  Contains code for feature extraction, preprocessing, classification, statistical analysis, and confusion metrics and ROC plot.
  
- `Italian Folder`  
  Contains code for Androids corpus feature extraction, preprocessing, classification, and confusion metrics and ROC plot.

- `English Folder`  
  Contains code for DAIC-WOZ and EDAIC feature extraction, preprocessing, classification,, and confusion metrics and ROC plot.
  
- `Cross-corpus Folder`  
  Contains code for cross-corpus and mixlingual experimets.

---

### Installation

To get started, follow the steps below.
### 1. Create a Conda environment with Python 3.10
<pre>
conda create -n myenv python=3.10
conda activate myenv
</pre>
### 2. Install dependencies:
After activating the environment, run the following command to install the required dependencies:
<pre>
pip install -r requirements.txt
</pre>

---
### 3. Datasets:
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



