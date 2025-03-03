**LLM Project:**
# Title here

## **Project Task**
(fill in details about your chosen project)

### Dataset
(fill in details about the dataset you're using)

### Deliverable
1.
2.

### Who benefits?
1. A
    - Who: 
    - Why: 
    - How: 
2. B
    - Who: 
    - Why: 
    - How: 
3. C
    - Who: 
    - Why: 
    - How: 

<br>

## **Process Overview**
![Process Visual. To follow.]()

### Pre-trained Models
(fill in details about the pre-trained model you selected)

### Performance Metrics
(fill in details about your chosen metrics and results)

### Hyperparameters
(fill in details about which hyperparameters you found most important/relevant while optimizing your model)

<br>

## **Reproducibility**
### file structure

### rig used deets

### env info & installation
1. Create Anaconda Env
    ```bash
    conda create -n ENV_NAME python=3.9
    conda activate ENV_NAME
    ```
2. Install dependencies
    ```bash
    conda install -c conda-forge jupyterlab scikit-learn tensorflow fastapi

    conda install -c anaconda ipykernel

    conda install pandas requests numpy scipy matplotlib seaborn nltk spacy gensim textblob

    conda install -c plotly plotly=5.24.1

    conda install transformers datasets torch
    
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    > Note: The `pytorch-cuda=11.8 -c nvidia` is a config specific to my rig. Look up on what would work best on your device.
3. Register Jupyter kernel

    `python -m ipykernel install --user --name=ENV_NAME`

### links, screenies, references