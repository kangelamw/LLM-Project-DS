**LLM Project:**
# Beyond Sentiment: Turning Negative Reviews into Actionable Insights

**Contents**
- [Project Details](#project-task)
- [Deliverable](#deliverable)
- [Who Benefits?](#who-benefits)
- [Process Overview](#process-overview)
- [Models & Datasets](#pre-trained-models--datasets)
- [Performance Metrics](#performance-metrics)
    - [phi-2's Initial Performance Summary](#phi-2s-initial-performance-summary)
    - [Fine-tuned phi-2's Performance Summary](#fine-tuned-phi-2s-performance-summary)
- [Hyperparameters](#hyperparameters)
- [Reproducibility](#reproducibility)
    - [File Structure](#repo-file-structure)
    - [Setting up](#setting-up-dependencies)
- [References (Readings, Models, etc.)](#references-readings-models-etc)

## **Project Task**
This project goes beyond traditional sentiment analysis by integrating a more complex emotion detection into review processing. Using fine-tuned LLMs and emotion classification, we analyze negative reviews to uncover deeper emotional context and generate actionable insights. Instead of simply labeling sentiment as “positive” or “negative,” this approach identifies why an experience went wrong and how to improve it, allowing businesses to address concerns more effectively with meaningful, constructive feedback.

### Deliverable
A model that can:
- Analyze sentiment and emotions in reviews.
- Turn complaints into useful, constructive insights.
- (If time allows) Provide an easy-to-use interface.

### Who benefits?
> You could fine-tune the final phi-2 trained model to your specific industry. I didn't do it here, but if you do it by batch, you could also get a summarization model to generate a summary of the outputs, and then you have your report.

1. Small & Medium Businesses (SMBs)
    - **Who:** Local restaurants, cafes, salons, retailers, hotels.
    - **Why:** They often lack dedicated data teams and automated feedback analysis tools.
    - **How:** Quickly identify common customer pain points; Get actionable suggestions without manually analyzing hundreds of reviews.

2. Online Marketplace Sellers
    - **Who:** Small e-commerce businesses that sell products on Etsy, eBay, Amazon, Shopify.
    - **Why:** These sellers rely on customer reviews but may lack the tools to help them generate insights from them.
    - **How:** Quickly extract common trends from reviews; Improve product quality based on insights.

3. Customer Support & Feedback Teams
    - **Who:** Customer service managers in larger companies.
    - **Why:** Many businesses rely on human agents to manually review feedback—this would automate and streamline that process effectively.
    - **How:** Suggest proactive responses to customer concerns.

<br>

### **Process Overview**
1. Environment Set-Up
2. Fetch and pre-process the dataset for inference.
3. Use SamLowe's RoBERTa-based emotion classification model on dataset.
4. Inspect and pre-process the dataset as input for output generation from a 'bigger' model: Mistral 7B
5. Finetune 'smaller' model: Phi-2 for inference & Evaluate.
6. Deployment
7. Interface Building.
8. Deployment


## Pre-trained Models & Datasets
Dataset &rarr; [YELP's Full Review Dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)

Pre-trained Model 01 &rarr; [TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- For text generation of constructive feedback to finetune phi-2 with.

Pre-trained Model 02 &rarr; [Microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- Finetune to generate constructive feedback on YELP reviews + emotions.

## Performance Metrics: BLEURT, BERTScore and METEOR
- **BLEURT:** best for capturing meaning
    <span style="font-size:12px;">
    |  | |
    |------------------|-------------|
    | > 0.5            | Very high similarity, high-quality generation |
    | 0.3 – 0.5        | Good similarity, but some minor differences |
    | 0.1 – 0.3        | Moderate similarity, some mismatches |
    | < 0.1            | Weak similarity, possible errors in generation |
    | < 0.0            | Very poor match, unrelated text |
    </span>
- **BERTScore:** best for semantic similarity (good for context-heavy text)
    <span style="font-size:12px;">
    | | |
    |--------------------|-------------|
    | > 0.9              | Almost identical output |
    | 0.8 – 0.9          | Very similar, minor differences |
    | 0.7 – 0.8          | Good similarity, noticeable differences |
    | 0.5 – 0.7          | Some overlap, but significant mismatches |
    | < 0.5              | Poor similarity, possibly incorrect output |
    </span>
- **METEOR:** best for matching n-grams & synonyms (useful for paraphrased text)
    <span style="font-size:12px;">
    | | |
    |------------------|-------------|
    | > 0.7            | Very high overlap, near-perfect match |
    | 0.5 – 0.7        | Good overlap, some variations |
    | 0.3 – 0.5        | Moderate similarity, but important differences |
    | 0.1 – 0.3        | Weak similarity, some keywords match but meaning differs |
    | < 0.1            | Poor similarity, almost no matching |
    </span>

<br>

> My current benchmark is mistral7b's generated responses to compare against non-fine-tuned phi-2's responses and the fine-tuned version's responses.

> It will serve as the 'ground truth' and I've read some of mistral's responses, they were pretty good. It could be better with some fine-tuning, but what we want is a more portable, smaller model.

---

### <center> phi-2's Initial Performance Summary </center>

<center>

|        |   bleurt |   bertscore |   meteor |
|:-------|---------:|------------:|---------:|
| mean   | -0.71545 |     0.84724 |   0.1909 |
| median | -0.67285 |     0.8481  |   0.1909 |

</center>

![bleurt](/images/bleurt_score_distribution.png)

![bertscore](/images/bertscore_f1_distribution.png)

Phi-2 is capturing the core meaning or context of the text well, suggesting it is understanding the essence of the task or response well (BERTScore F1). Despite the good contextual understanding, Phi-2 is not aligning well with Mistral’s output at the meaning level (BLEURT Score). This could indicate that it’s missing some subtle semantic nuances, or not adhering strictly to the expected response as shown below (METEOR score against BLEURT and BERTScore)

![scatterplot](/images/phi-2_vs_mistral_alignment.png)

You can read more about it [here](/notebooks/3-pre-trained-model.ipynb).

| initial_metrics |   count |      mean |         std |     min |      25% |      50% |       75% |    max |
|:----------|--------:|----------:|------------:|--------:|---------:|---------:|----------:|-------:|
| bleurt    |    1000 | -0.756606 | 0.353658    | -1.6119 | -1.02372 | -0.70085 | -0.488175 | 0.8601 |
| bertscore |    1000 |  0.838373 | 0.0194086   |  0.787  |  0.8235  |  0.8395  |  0.852025 | 0.9851 |
| meteor    |    1000 |  0.2222   | 5.22066e-15 |  0.2222 |  0.2222  |  0.2222  |  0.2222   | 0.2222 |

---
---
---

### <center> Fine-tuned phi-2's Performance Summary </center>

<center>

|        |    bleurt |   bertscore |   meteor |
|:-------|----------:|------------:|---------:|
| mean   | -0.376466 |    0.885482 |   0.2318 |
| median | -0.33835  |    0.8873   |   0.2318 |

</center>

![fine_bleurt](/images/fine-tuned_bleurt_score_distribution.png)

![fine_bertscore](/images/fine-tuned_bertscore_f1_distribution.png)

blaaah

![fine_scatterplot](/images/fine-tuned_phi-2_vs_mistral_alignment.png)

| fine-tuned metrics  |   count |      mean |         std |     min |       25% |      50% |       75% |    max |
|:----------|--------:|----------:|------------:|--------:|----------:|---------:|----------:|-------:|
| bleurt    |    1000 | -0.376466 | 0.286724    | -1.8587 | -0.548975 | -0.33835 | -0.159225 | 0.2163 |
| bertscore |    1000 |  0.885482 | 0.0166502   |  0.7906 |  0.876675 |  0.8873  |  0.897    | 0.9282 |
| meteor    |    1000 |  0.2318   | 1.44401e-15 |  0.2318 |  0.2318   |  0.2318  |  0.2318   | 0.2318 |

<br>


## Hyperparameters
#### During training:
- learning_rate=2e-5
    - 
- metric_for_best_model="loss" + greater_is_better=False
    - minimizing loss
- num_train_epochs=3
    - any more and it starts overfittinh
- weight_decay=0.01
- optim="adamw_torch_fused"

#### During generation:
- temperature=0.5
    - the model tends to tell stories or write entirely different things
- top_p=0.9
    - stricter generation
- no_repeat_ngram_size=3
    - it was a bit of a parrot without this

<br>

## **Reproducibility**
### Repo File Structure

    ├── data
    │   ├── // The rest were not included/uploaded, more like data checkpoints //
    │   ├── ready_for_phi-2
    │   │   ├── eval
    │   │   │   ├── fine-tuned_phi2_vs_mistral_scores.csv
    │   │   │   ├── initial_phi2_vs_mistral_scores.csv
    │   │   │   ├── train10k_sample.csv
    │   │   │   ├── train10k_sample_fine-tuned.csv
    │   │   ├── test_01.csv
    │   │   ├── test_02.csv
    │   │   ├── train_01.csv
    │   │   ├── train_02.csv
    ├── images
    │   ├── // images //
    ├── models
    │   ├── // checkpoints //
    │   ├── phi-2_01
    │   │   ├── // model path //
    │   ├── phi-2_full_2
    │   │   ├── // Model v2 -- evaluated //
    │   ├── runs
    │   │   ├── // RUNS //
    ├── notebooks
    │   ├── 1-preprocessing.ipynb
    │   ├── 2-representation.ipynb
    │   ├── 3-pre-trained-model.ipynb
    │   ├── 4-optimization.ipynb
    │   ├── 5-deployment.ipynb
    │   ├── etc.ipynb
    │   ├── functions.py
    ├── Project_Journal.md
    ├── README.md

### Rig:
- **GPU:** 12GB NVidia RTX 3060TI
- **RAM:** 16GB

### Setting up Dependencies
1. Create Anaconda Env
    ```bash
    conda create -n ENV_NAME python=3.9
    conda activate ENV_NAME
    ```
2. Install dependencies
    ```bash
    conda install -c conda-forge jupyterlab scikit-learn tensorflow fastapi joblib fastparquet pyarrow pillow

    conda install -c anaconda ipykernel

    conda install pandas requests numpy matplotlib seaborn nltk textblob ipywidgets

    conda install -c plotly plotly=5.24.1

    conda install transformers datasets accelerate peft evaluate
    
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # Check the official pytorch page for installations for your device/versions.
    
    ```
3. Register Jupyter kernel

    `python -m ipykernel install --user --name=ENV_NAME`


### References (Readings, Models, etc.)
*currently collecting these in [project journal](/Project_Journal.md)*