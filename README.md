This is a project made to spot and detecty fashion trends on reddit. I scrapped over 15,000 image containing posts from multiple subreddits on reddit using PRAW. Used CLIP model by OpenAI to classify these images into their respective classes (cloth type). Carried out sentiment analysis over their titles and body texts to understand if the post if in positive or negative context for the posted photo. Calculated Impact via various mathematical methods and plotted the trend of these clothing items for the past two weeks. 
I also bulit a ResNet model replica in the dream of training it myself and using it in my project, but my laptop would need 2 months of non stop compute to train the resnet50 model one a decently sized dataset, so i gave up the dream and used CLIP as it has more classes for fashion clothing as compared to resnet50.

Trends Plotted Via Different Impact Calculations:

![Bayesian Impact](Sentiment%20-%20Trend%20Bayesian.png)

![Time Decay Impact](Sentiment%20-%20Trend%20Time%20Decay.png)