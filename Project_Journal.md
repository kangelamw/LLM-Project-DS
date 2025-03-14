# Notes & Scribbles
This may be a bit of a wall. Just a bunch of thought processes. It's semi-structured at best. It's sub-thoughts with sub-thoughts. A dendogram, but new branches can appear sporadically.

**<span style="font-size:18px;">Table of Contents</span>**
- [Project Details in steps](#planning)
- [Thoughts & Ideas](#thoughtsideas)
  - [Pre-trained Model by Sam Lowe](#found-a-pre-trained-model-by-samlowe)
  - [Time Constraints](#time-constraints)
  - [Final Function STructure](#final-function-structure)
  - [Build Interface](#build-interface----flutterflow-it-allows-api-calls)
- [Resources](#resources)

---

<br>

## Planning
| Process |                                                       | Status   |
|----|------------------------------------------------------------|----------|
| 01 | Environment Set-Up                                         | Completed, Day 01 |
| 02 | Post project details on Discord                            | Completed, Day 01 |
| 03 | Inspect, Clean & Prep: YELP Dataset                        | Completed, Day 02 |
| 04 | -- Tokenize: YELP Dataset --                               | Completed, Day 02 |
| 05 | Check with a mentor: Compass AR                            | Completed, Day 03 |
| 06 | Classify YELP Dataset                                      | Completed, Day 03 |
| 07 | Inspect & Prep New Dataset                                 | Completed, Day 03 |
| 08 | Generate outputs from mistral-7b to fine-tune phi-2 with.  | Completed, Day 08 |
| 09 | Prep/Train phi-2                                           | Pending |
| 10 | Evaluate model                                             | Pending |
| 11 | Build final function to generate output.                   | Pending |
| 12 | Deploy & Test API                                          | Pending |
| -- | **Create Interface *(If we have at least 2-3 days left)*** | --- --- |
| 01 | Build Flutterflow UI                                       | Pending |
| 02 | Test/Connect API                                           | Pending |
| 03 | Deploy & Test Web App                                      | Pending |

---

<br>

## Thoughts/Ideas
- On the Yelp Dataset...
    - There were reviews being categorized as `class: float` and it broke the inference during the emotion classification phase. Upon further inspection, they were reviews with no words at all, just special characters/punctuations&rarr;These rows were removed. There was 13.
- How would I output emotions if the main model won't be necessarily linked to the first one when running its own inference? That's not...not doable. Inside the function for the final output... call the classification function, and then run on mistral. This way I can output both emotion and feedback.
    - I'll skip the part where it does batch processing of CSVs in the web app for now. I had to run the 1st inference on our data overnight. It will not do great on a 5 min demo.
    - It has to pre-process the review/clean it on input. This is gonna be one long py file.
    - It doesn't have to see that again, the output could just be... emotions gathered + constructive feedback. Would be nice if it could do batch processing, but that can be compute intensive depending on how much data they have.
- Mistral has 7B params, i'll try to find a smaller one to finetune. SamLowe's has 125M.
- I'm gonna use Microsoft's phi-2 model. It only has 1.1B params, it's a lot smaller and may be a better fit for our task. I've read in passing somewhere, that MS phi's models are great for logic reasoning. I think it fits better with our business usecase. Below is a comparison between mistral-7b and phi-2:

| Feature        | Mistral 7B         | Phi-2 (2.7B)         |
|---------------|--------------------|----------------------|
| **Size**      | 7 billion params   | 2.7 billion params  |
| **Architecture** | Dense Transformer | Dense Transformer  |
| **Context Length** | 8K tokens | 4K tokens |
| **Optimized for?** | General NLP, reasoning | Code & reasoning |
| **Hardware Requirements** | Needs **>24GB VRAM** for full precision, but can be quantized to fit. | Runs on **8-12GB VRAM** with quantization |
- I went back and read up on phi-2 [on their page](https://huggingface.co/microsoft/phi-2). They say it showcased a state-of-the-art performance among models with less than 13B params.

- Before I could use the models, I had to turn on developer mode on windows. Just go to settings and search for it, [or read it for yourself here](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development).

- Also had to install [CUDA from NVIDIA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) to be able to actually use my gpu. I clued in when my RAM was screaming at 98% while the GPU was just chilling at 0%.

- For training time, I try to do 'ground work' during the day, typing out the code, reading up on what to do, scribbling away on this journal. I start running/testing the inferences at night where it can do the full inference uninterrupted by random daytime browsing. I check on it every 1-2 hours through the night, it's done by morning. I try to get the bugs out of the way before running the full thing, it's mainly checking if the PC is still on and the cell is still running&mdash;because cats.

- I still have a use for mistral-7b, but instead of finetuning it, it'll just be the 'big model to learn from' for phi-2. I think a smaller model to run the final output with would be better for performance. I think the goal of finetuning is taking a model beyond its general capabilities for our specific task. This is a good experiment.

- Mistral's quantized model isn't going great. I can't seem to find any info about running it on the GPU, it may not even support it. I've spent enough time (24h) on getting it to work. I'll use the mistral base model and quantize it myself&mdash;**It took me days, but it WORKED**

- I'm not sure how to evaluate it yet. There's a module called 'judges' for llm outputs. I'll read up on how it could be utilized for this project when I'm closer to that step/while waiting for inference to finish. A lot of this project is learning it as I go...

- I've read through a couple of papers (referenced below) on ways of evaluating a model's qualitative outputs. I've thought of thematic analysis, which i thought was a fancy name for topic modeling + classification. I've done this before during my undergrad, but by hand. It was not fun. It also wouldn't be as relevant to this current step in the project where I'm getting phi-2 trained on mistral's responses.

- phi-2 was initialized and tested (if it was working/loaded properly) and so far, not looking very promising. I ran a few prompts the same ones that mistral got, and phi-2 was trying to tell me the story of the business owner and how they felt about their cooking getting ridiculed by the customer. Hilarious, but not what I want.

<br>

### Found a pre-trained model by SamLowe
I found [this pre-trained model by Sam Lowe](https://huggingface.co/SamLowe/roberta-base-go_emotions). It was trained on RoBERTa as well, but I debated not to use it. I find 27 emotions to be too complex, as it's not going to be my final output. I can remove, combine/merge specific emotions to reduce redundancy and to tailor the dataset to our purposes more effectively. The dataset was from social media(Reddit), so it may be a bit more emotionally nuanced than reviews.

From [GoEmotions README](https://github.com/google-research/google-research/blob/master/goemotions/README.md):
```
GoEmotions contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral.

The emotion categories are:
_admiration, amusement, anger, annoyance,
approval, caring, confusion, curiosity,
desire, disappointment, disapproval, disgust, 
embarrassment, excitement, fear, gratitude, 
grief, joy, love, nervousness,
optimism, pride, realization, relief, 
remorse, sadness, surprise_.
```

**<center>Thoughts:</center>**

> Maybe I'll drop the rows with specific labels... I need to see the datasets first and see how many rows I'd be left with to train from if I do this. Since this dataset was carefully curated by Google Researchers, it may be well-balanced in terms of class. I'll definitely disrupt that balance when I do these changes.

To combine/merge/drop one:
  - admiration vs approval &rarr; **approval**
    > <span style="font-size:12px;">in the same spectrum, we only need one&mdash;approval is more relevant</span>
  - disapproval vs disappointment &rarr; **disapproval**
    > <span style="font-size:12px;">both conveys dissatisfaction, we only need one&mdash;disapproval is more relevant</span>
  - fear + nervousness &rarr; **anxiety**
    > <span style="font-size:12px;">basically in the same spectrum, we don't need both</span>
  - grief vs sadness &rarr; **sadness**
    > <span style="font-size:12px;">grief may be too emotionally complex for our model</span>
  - embarrassment + remorse = **regret** 
    > <span style="font-size:12px;">both involves wishing something had gone differently&mdash;regret will capture both</span>
  - amusement vs surprise &rarr; **surprise**
    > <span style="font-size:12px;">amusement is always positive, surprise can be positive or negative&mdash;when combined with another emotion, should provide us with deeper context</span>
  - optimism + pride = **confidence**
    > <span style="font-size:12px;">optimism is believing things will go well, and pride is feeling good about achievements&mdash;confidence captures both.</span>
  - caring + love = **affection**
    > <span style="font-size:12px;">interchangeable in most cases</span>

To drop:
  - realization &rarr; more of a cognitive process than an emotion
  - relief &rarr; more like the absense/removal of a negative emotion

<br>

What's left of the 27:
1. Approval
2. Disapproval
3. Anxiety
4. Sadness
5. Regret
6. Surprise
7. Confidence
8. Affection
9. Anger
10. Gratitude

<br>

### Time constraints
Day 2, I decided to just use SamLowe's model and save me about 3-4 days of cleaning/pre-processing, training, etc. Makes up for some time to do the interface or finetuning mistral.

I can still go back and try to apply this later, but the research/readings I'd need to do for this might take a day or two + actually doing it. I'll go back if I have time.

I *may* have mis-stepped and tried generating 650k (train rows) to train phi-2 with. For fine-tuning, I've read that 5-10k should be enough, I sampled the data accordingly. (Stratified) I have a total of 15k to train with and 3k to test with. I'm gonna train **as needed** with 2.5k at a time.

### Final function structure
Input:
- label (number of stars)
- review
  - needs cleaned, so call the text cleaning function
- emotions
  - needs to call SamLowe's model

To turn into prompt... call the prompt creation function
- call phi-2 for generation
- clean output as needed
- return the feedback and emotion gathered

I created a whole section for this, I thought it was gonna be harder than this to mentally process. Not really, but now I've written down a step-by-step structure to follow and go back to.

<br>

### Build Interface -- Flutterflow... It allows API calls.
- UI
- Enter review...
	- Output emotion classification result.
	- Output generated insights.
- Analyze btn, linked to function
- Loading indicators
- Output csv with... review + emotion + constructive feedback
  > - Then the output can be reviewed for accuracy by a human before running with it.
  >   - Explainability of output&mdash;*Did the model do goodd?*
  > - Feedback output can be processed further/summarized.
  > - Emotion classification is retained, get an overview of HOW the customers generally feel about the service.

I started this on day 08. With FlutterFlow, it's not that hard to build the UI. Just really trying to fill the time while inference is taking its time, I've optimized it the best way I was able to do. It's a waiting game.

Finished the UI in about 2-3 hrs.

<br>

### Evaluating the model
I am debating trying BLEURT, BERTscore and MeteorScore. All are non-LLM scorer according to [this](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation) article.

- We are using:
  - [BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)](https://github.com/google-research/bleurt)
  - BERTScore
  - Meteor Score -- Not using due to (seemingly endless) dependency errors...

  <span style="font-size:12px;">

  | **BLEURT Score** | **Meaning** |
  |------------------|-------------|
  | > 0.5            | Very high similarity, high-quality generation |
  | 0.3 – 0.5        | Good similarity, but some minor differences |
  | 0.1 – 0.3        | Moderate similarity, some mismatches |
  | < 0.1            | Weak similarity, possible errors in generation |
  | < 0.0            | Very poor match, unrelated text |

  <br>

  | **BERTScore (F1)** | **Meaning** |
  |--------------------|-------------|
  | > 0.9              | Almost identical output |
  | 0.8 – 0.9          | Very similar, minor differences |
  | 0.7 – 0.8          | Good similarity, noticeable differences |
  | 0.5 – 0.7          | Some overlap, but significant mismatches |
  | < 0.5              | Poor similarity, possibly incorrect output |

  <br>

  | **METEOR Score** | **Meaning** |
  |------------------|-------------|
  | > 0.7            | Very high overlap, near-perfect match |
  | 0.5 – 0.7        | Good overlap, some variations |
  | 0.3 – 0.5        | Moderate similarity, but important differences |
  | 0.1 – 0.3        | Weak similarity, some keywords match but meaning differs |
  | < 0.1            | Poor similarity, almost no matching |

  </span>

> It was mentioned that these may be unreliable, but a combination of them seems to be doing the trick.


> I did consider using `judges`, but the dependencies were c r a z y.

---

<br>

# Resources
### <span style="font-size:22px;">Readings</span>
> Just random things I found that I read to inform decisions, but not all made it to this project.

**Enable your device for Development**
-   https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

**GoEmotions**
- https://github.com/google-research/google-research/blob/master/goemotions/README.md
- https://huggingface.co/datasets/google-research-datasets/go_emotions

**Leveraging LLM-as-a-Judge for Automated and Scalable Evaluation**
- https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method
- https://github.com/confident-ai/deepeval

**Reliable Confidence Intervals for Information Retrieval Evaluation Using Generative A.I.**
- https://dl.acm.org/doi/10.1145/3637528.3671883

**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**
- https://arxiv.org/abs/2306.05685

**BLEURT: a Transfer Learning-Based Metric for Natural Language Generation**
- https://github.com/google-research/bleurt

---

### <span style="font-size:22px;">Models</span>
> Used in the project

**SamLowe/roberta-base-go_emotions**
- https://huggingface.co/SamLowe/roberta-base-go_emotions
- https://github.com/samlowe/go_emotions-dataset/blob/main/eval-roberta-base-go_emotions.ipynb

**Microsoft/phi-2**
- https://huggingface.co/microsoft/phi-2

**Mistral-7B-v0.1**
- https://huggingface.co/mistralai/Mistral-7B-v0.1

**judges**
- https://pypi.org/project/judges/

<br>

---

### <span style="font-size:22px;">Datasets</span>
> Base dataset used in the project.

**YelpReviewFull**
- https://huggingface.co/datasets/Yelp/yelp_review_full
- Citation Information
  - Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015)