# Out-of-distribution detection for deep learning DNA barcoding

Codes in this repository are for performance evaluation of deep learning models for classification/identification and out-of-distribution detection of DNA barcoding. Out-of-distribution (OOD) samples are samples from the classes which are NOT present in the training dataset. For example, when a model is trained to identify DNA sequences of cat and dog, a sequence of human, or of any other species which are neigher cat nor dog, is an OOD sample. Presense of OOD samples leads a deep learning model to an erroneous identification if they are not handled properly. 

The model detects such OOD samples by calculating metrics for OOD detection. Currently, we implement simple three metrics, softmax probability, energy score and Mahalanobis distance score.  

The details of the studies are presnented in the manuscript available at the following link.

http://linktothemanuscript


## Requirments:
* Keras & Tensorflow (Only Keras 2.XX. Keras >= 3.0 is not supported)
* scikit-learn
* pandas
* Biopython 

Library versions used in analyses in the manuscript is listed in requirements.txt.

## Creating an environment with conda:
If you are not familiar with how to manage environments, using Anaconda is probablyt the easiest option.
https://www.anaconda.com/

After installing Anaconda, use the following commands in Anaconda Prompt to create an environment for running the codes.

```
conda creat -n barcoding_cnn python=3.10 

conda activate barcoding_cnn
```

Then, install required libraries with pip.
**Folloing commands must be run in the activated conda environment**, or you may destroy your environment.
```
pip install -r requirements.txt
```
or 
```
python -m pip install -r requirements.txt
```
To exit from the environment...
```
conda deactivate 
```
Make sure to activate the created environment before running the codes below.

## Code usages:
### Python script files
#### train_cnn_all_length.py :
```
python train_cnn_all_length.py in.fa sp.table.in.txt ood.fa sp.table.ood.txt run_code
```
This code replicates the analyses conducted in the manuscript. Training & testing processes are repeated 20 times for each parameter combination. Results (identification results and OOD scores) are written in files pred.prob.metrics.[run_code].txt and pred.prob.metrics.ood.[run_code].txt. The first file contains results of in-distribution data and the second one, of ood data.

- `in.fa` : a Fasta alignment file of in-distribution(ID) samples
- `sp.table.in.txt` :  a comma separated text file for ID sample labels

- `ood.fa` : a Fasta alignment file of out-of-distribution (OOD) samples
- `sp.table.ood.txt` : a comma separated text file for OOD sample labels

- `run_code`: string to specify an output file prefix

#### calc_stats.py :
```
python calc_stats.py pred.prob.txt pred.prob.ood.txt run_code
```
This code calculates performance metrics such as accuracy and false negative rate, from output files created above. It writes down performance metrics in a file, stat.[run_code].txt. 

- `pred.prob.txt`: an output file of train_cnn_all_length.py for ID samples
- `pred.prob.ood.txt`: an output file of train_cnn_all_length.py for OOD samples
- `run_code`: string to specify an output file prefix

#### train_save_model.py : 
```
python train_save_model.py in.fa sp.table.in.txt ood.fa sp.table.ood.txt run_code
```
This code trains and saves models for barcoding classification. Saved models are later used to run GradCAM-related codes or other applications. Models are saved in files, model.[run_code].{650,300,150}.keras. Models for three fragment lengths are saved.
Input files are identical to the ones for train_cnn_all_length.py.

#### grad_cam.py and grad_cam_energy.py:

These codes conduct gradient-based attribution of model decisions. grad_cam.py calculates GradCAM scores for 8bp windows, which localize regions of importance for classification. grad_cam_energy.py calculates similar scores for energy-based OOD detection, which can find regions contributing to OOD-ness.
```
python grad_cam.py runcode.650.keras in.fa
```
The above code calculate GradCAM scores for in.fa with a model, runcode.650.keras
```
python grad_cam_energy.py runcode.650.keras ood.fa
```
This time grad-energy scores for ood samples are calculated. 


### Data files:
The "data" folder contains a data set for testing. They are sequence alignments in Fasta format and class labels in comma-delimited texts.

- drosophila_NA.15sp.in.fa : Fasta alignment file of in-distribution samples
- drosophila_NA.49sp.out.fa : Fasta alignment file of out-of-distribution samples

- sp.table.drosophila_NA.in.15sp.txt : a comma separated text file for ID sample labels
- sp.table.drosophila_NA.out.49sp.txt : a comma separated text file for OOD sample labels

Full datasets are available for download at Dryad data repositry. (link)

