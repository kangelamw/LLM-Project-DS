# Notes & Scribbles

## Etc.
I found [this](https://huggingface.co/SamLowe/roberta-base-go_emotions) pre-trained model by Sam Lowe. It was trained on RoBERTa as well, but I decided not to use it. I find 27 emotions to be too complex, as it's not going to be my final output. I can remove, combine/merge specific emotions to reduce redundancy. Thoughts:

Combine/Merge/Drop one:
  - admiration vs approval &rarr; **approval**
  - disapproval vs disappointment &rarr; **disapproval**
    > both conveys dissatisfaction, we don't need both... disapproval is more relevant to our purposes.
  - fear + nervousness &rarr; **anxiety**
    > basically in the same spectrum, we don't need both
  - grief vs sadness &rarr; **sadness**
    > grief may be too deep of an emotion for our purposes
  - embarrassment + remorse = **regret** 
    > both involves wishing something had gone differently
  - amusement vs surprise &rarr; **surprise**
    > amusement is always positive, surprise can be positive or negative, when combined with another emotion, should us provide deeper context
  - optimism + pride = **confidence**
    > optimism is believing things will go well, and pride is feeling good about achievements... confidence captures both.
  - caring + love = **affection**
    > interchangeable in most cases

Drop:
  - realization &rarr; more of a cognitive process than an emotion
  - relief &rarr; more like the absense/removal of a negative emotion

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

## Resources
