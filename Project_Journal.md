# Notes & Scribbles
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
| 08 | Get Claude API access; Create new Dataset                  | Pending |
| 09 | Prep/Train Mistral-7B                                      | Pending |
| 10 | Evaluate model                                             | Pending |
| 11 | Deploy & Test API                                          | Pending |
| -- | **Create Interface *(If we have at least 2-3 days left)*** | --- --- |
| 01 | Build Flutterflow UI                                       | Pending |
| 02 | Test/Connect API                                           | Pending |
| 03 | Deploy & Test Web App                                      | Pending |

---

<br>

### Thoughts/Ideas


### Etc.
1. On the Yelp Dataset...
    - There were reviews being categorized as `class: float` and it broke the inference during the emotion classification phase. Upon further inspection, they were reviews with no words at all, just special characters/punctuations&rarr;These rows were removed. There was 13.
2. How would I output emotions if the main model won't be necessarily linked to the first one when running its own inference? That's not...not doable. Inside the function for the final output... call the classification function, and then run on mistral. This way I can output both emotion and feedback.
    - I'll skip the part where it does batch processing of CSVs in the web app for now. I had to run the 1st inference on our data overnight. It will not do great on a 5 min demo.
    - It has to pre-process the review/clean it on input. This is gonna be one long py file.
    - It doesn't have to see that again, the output could just be... emotions gathered + constructive feedback. Would be nice if it could do batch processing, but that can be compute intensive depending on how much data they have.
3. Mistral has 7B params, i'll try to find a smaller one. SamLowe's has 125M.

<br>

## Found a pre-trained model by SamLowe
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

## Time constraints
Day 2, I decided to just use SamLowe's model and save me about 3-4 days of cleaning/pre-processing, training, etc. Makes up for some time to do the interface or finetuning mistral.

## Build Interface -- Flutterflow... It allows API calls.
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

## Resources
