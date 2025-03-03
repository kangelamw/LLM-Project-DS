# Notes & Scribbles
## Thoughts/Ideas


## Planning
| Process |                                                       | Status   |
|----|------------------------------------------------------------|----------|
| 01 | Environment Set-Up                                         | Complete |
| 02 | Post project details on Discord                            | Complete |
| 03 | Inspect, Clean & Prep: YELP Dataset                        | Pending |
| 04 | -- Tokenize: YELP Dataset --                               | Pending |
| 05 | Check with a mentor: Compass AR                            | Pending |
| 06 | Classify YELP Dataset                                      | Pending |
| 07 | Inspect & Prep New Dataset                                 | Pending |
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

## Found a pre-trained model by SamLowe
I found [this pre-trained model by Sam Lowe](https://huggingface.co/SamLowe/roberta-base-go_emotions). It was trained on RoBERTa as well, but I decided not to use it. I find 27 emotions to be too complex, as it's not going to be my final output. I can remove, combine/merge specific emotions to reduce redundancy and to tailor the dataset to our purposes more effectively. The dataset was from social media(Reddit), so it may be a bit more emotionally nuanced than reviews.

From [GoEmotions README](https://github.com/google-research/google-research/blob/master/goemotions/README.md):
```
GoEmotions contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral.

The emotion categories are:
_admiration, amusement, anger, annoyance, approval,
caring, confusion, curiosity, desire, disappointment,
disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness,
optimism, pride, realization, relief, remorse,
sadness, surprise_.
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
I'll just use SamLowe's model and save me about 3-4 days of cleaning/pre-processing, training, etc. Makes up for some time to do the interface.

## Build Interface -- Flutterflow... It allows API calls.
- UI
- Upload csv...?
	- Add columns: Emotions
	- Add columns: Generated insights for each review
- Analyze btn
- Loading screen!
- Output csv with... review + emotion + constructive feedback
  > - Then the output can be reviewed for accuracy by a human before running with it.
  >   - Explainability of output&mdash;*Did the model do goodd?*
  > - Feedback output can be processed further/summarized.
  > - Emotion classification is retained, get an overview of HOW the customers generally feel about the service.

## Resources
