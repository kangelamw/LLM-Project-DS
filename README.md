**LLM Project:**
# Beyond Sentiment: Turning Negative Reviews into Actionable Insights

## **Project Task**
This project goes beyond traditional sentiment analysis by integrating a more complex emotion detection into review processing. Using fine-tuned LLMs and emotion classification, we analyze negative reviews to uncover deeper emotional context and generate actionable insights. Instead of simply labeling sentiment as “positive” or “negative,” this approach identifies why an experience went wrong and how to improve it, allowing businesses to address concerns more effectively with meaningful, constructive feedback.

### Deliverable
A model that can:
- Analyze sentiment and emotions in reviews.
- Turn complaints into useful, constructive insights.
- (If time allows) Provide an easy-to-use interface.

### Who benefits?
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

## **Process Overview**
![Process Visual. To follow.]()

### Pre-trained Models, Datasets & API
#### Dataset: [YELP's Full Review Dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)
- The page implies it may contain non-english reviews, so I'll try and keep just english ones.

#### API: [Claude]() ! GET LINK !
- Generate constructive feedback on YELP reviews + emotions to train mistral-7b with.

#### Pre-trained Model: [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) w/ QLoRA
- For text generation of constructive feedback

### Performance Metrics
(fill in details about your chosen metrics and results)

### Hyperparameters
(fill in details about which hyperparameters you found most important/relevant while optimizing your model)

<br>

## **Reproducibility**
### Repo File Structure
>> Later

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
    conda install -c conda-forge jupyterlab scikit-learn tensorflow fastapi joblib fastparquet pyarrow

    conda install -c anaconda ipykernel

    conda install pandas requests numpy scipy matplotlib seaborn nltk spacy gensim textblob ipywidgets

    conda install -c plotly plotly=5.24.1

    conda install transformers datasets
    
    conda install pytorch torchvision torchaudio torch pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    > Note: The `pytorch-cuda=11.8 -c nvidia` is a config specific to my rig. Look up on what would work best on your device.
3. Register Jupyter kernel

    `python -m ipykernel install --user --name=ENV_NAME`

### links, screenies, references