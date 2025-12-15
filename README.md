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
To fully reproduce our results reported in the paper, please follow the below steps: 
### Installation

To get started, follow the steps below.
### 1. Create a Conda environment with Python 3.10
<pre>
conda create -n myenv python=3.10
conda activate myenv
</pre>
### 2. Clone the reop:
After activating the environment, run the following command to clone this repo:
<pre>
https://github.com/bubaimaji/MLDD.git
</pre>
### 3. Install dependencies:
After cloning this repo in your environment, run the following command to install the required dependencies:
<pre>
pip install -r requirements.txt
</pre>

### 2. Run:
To reproduce our results reported in the paper on the Bengali language, please follow the below steps: 
<pre>

</pre>


---
### 3. Datasets:
----
### Indic-Bengali
The Indic-Bengali is a dataset consist of audio files of 70 subjects (30 depressed and 40 healthy individuals). To access the dataset, please fill out this below form to get the Google Drive link:

- <a href="https://docs.google.com/forms/d/e/1FAIpQLSckj6z6tdl63eZCj-gPhpRYCuCpYRC3ybta56Xq_DjpdxCJzA/viewform?usp=publish-editor" target="_blank" rel="noopener noreferrer">Request Dataset Access</a>

### Form Requirements:
`Name`, `University`, `Department`, `Email address`, and `Ethical consent agreement` 

After submitting the form with consent, you will receive the Google Drive link via email within a few minutes.

### How to use:

Dataset used 5-fold speaker independent.

Each fold contains unique speaker data.

bangla_5fold_metadata.csv file contains meta data

------
### DAIC-WOZ and EDAIC
These two data can be downloaded from here: [DAIC-WOZ and EDAIC](https://dcapswoz.ict.usc.edu/). 

------
### Androids-Corpus (Italian)
Download the data from here: [Androids-Corpus](https://github.com/androidscorpus/data) dataset. 



