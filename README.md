# PREDICTING DENGUE FEVER OUTBREAKS

## Overview
Dengue Fever is a disease with severity ranging from flu-like symptoms to low blood pressure and death. Humans with Dengue Fever are not contagious; it can only be spread by Mosquitoes. It is typically observed in tropical regions, but cases have increased significantly in recent years; Health Officials and scientists are warning that climate change is likely to produce shifts that enable mosquitos to reach and infect new regions of the world.

## Hypothesis
Dengue Fever is spread by mosquitoes, whose breeding patterns are related to weather patterns. Therefore, the hypothesis of this competition is that an analysis of weather patterns can help predict outbreaks of the disease.

The competition provides detailed weather data for two tropical cities: San Juan, Puerto Rico (1990-2008), and Iquitos, Peru (2000-2010).

## Exploratory Data Analysis & Pre-Processing
If the weather drives outbreaks, one thing we'd expect to see is outbreak seasonality.

In order to visualize this, I first had to correct some date oddities. Because the week numbers given were calculated with an ISO standard, some years had the first week labelled as week 53, the second week labelled as week 1, the third week labelled as week 2, etc. ending in week 51. This was fixed by incrementing all week numbers for those years, then reducing any week 54's to week 1.

It does appear that there is some outbreak seasonality in these cities:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/e23adb70-9e6b-40cf-b5ba-0528e6655d70)

The data includes 20 weekly features that loosely fall into these categories:

- Temperature: Maximums, Minimums, Averages, and Ranges
- Precipitation
- Humidity
- Satellite Vegetation

At first glance, none of them appear to be strongly correlated with Dengue Cases:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/fcf85dfb-91b1-436b-a07a-123ce8c7e4c2)

But Pearson correlations do not consider sequences; this only means that there are no strong relationships in the current period. In other words, this means a temperature change in week X does not correlate to reports of Dengue Fever in Week X. It says nothing about reports of Dengue Fever in Week X+1, or Week X+2.

Indeed, a seasonal look at several features (such as average temperature, below) is suggestive of sequential relationships:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/e8cc196b-273d-4b63-908f-bb4b7c29d091)

Before starting any time series techniques, we need to have some idea how far back in time a model should consider when making predictions. Let's study the domain of Dengue Fever a bit and make our hypothesis more specific:

### What chain of events is most conducive to an outbreak of Dengue Fever?
I broke this question into 3 parts:

1. **What weather conditions are ideal for mosquitoes to breed and hatch?**
   The most common species of mosquito that spreads Dengue Fever, is Aedes aegypti.

   Research published in Oecologia [1] and in the Journal of the American Mosquito Control Association [2], suggests that temperature plays an important role in the viability of eggs; they have evolved best for hot and humid climates. Like all mosquitoes, their eggs need water to hatch. However, unlike other species' eggs, aegypti eggs can sit dormant in dry areas for a long time. Many sites (example) suggest that in dry seasons, aegypti eggs can survive over a year without moisture, and still hatch successfully once exposed to water. Since Dengue Fever is passed from mother to offspring; this is a huge reason why Dengue Fever is so persistent.

   A leading indicator for large egg hatching events might therefore be long dry periods followed by large rainfalls, or extended wet/humid seasons.

2. **How long does it take a hatched mosquito to reach a contagious life phase?**
   Exposed to water in a warm climate, aegypti eggs develop in as little as two days.

   Subsequent larvae development depends on the temperature. In their ideal climates, the larvae stage can last about a week. (Once again, during cooler periods, larvae with water access can survive for months.)

   Larvae then enter the Pupae stage, which lasts about 2 days. At that point, the mosquito is an adult, and the females are able to spread disease - approximately 8-10 days from the eggs being laid.

3. **How long after infection do Dengue Fever symptoms appear?**
   According to the CDC and the Mayo Clinic, symptoms start 3-7 days after being bitten by an infected mosquito.

   My hypothesis is that given wet weather conditions following a dry period, outbreaks can occur within 2 weeks and the reports of Dengue Fever cases can be expected within 3 weeks.

   ![image](https://github.com/Monish24/deng-ai-master/assets/54630644/65f78115-7c7f-4efe-b0ed-6061af15eb6c)

## Feature Selection
Based on the above domain knowledge and further exploratory analysis (available at my github), I decided not to use some of the features, including all the vegetation index values, and selected key temperature, humidity, and rain variables for each city. For each of the techniques, I used at least 3 weeks of data to make any prediction about outbreaks.

**San Juan:**
- Maximum Temperature
- Mean of (minimum temperatures, dew point temperature, and air temperature)
- Relative Humidity Percent
- Specific Humidity g/kg
- Precipitation kg/m^2

**Iquitos:**
- Mean Temperature
- Minimum Air Temperature
- Dew Point Temperature
- Temperature Range
- Specific Humidity g/kg
- Precipitation amount (mm)

In addition, I created boolean fields for the key seasons in each city, using the black line cutoffs noted below:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/c44d2574-d3a5-4256-8de3-7466f07b0b91)

## Time Series Techniques
I explored 2 types of time series solutions:

1. Long Short Term Memory models (LSTM neural networks)
2. Supervised Learning models with lagged features

## LSTM Training
As with all neural networks, best practice is to train an LSTM with training data and assess it using the loss on a validation set. Most use cases allow for random holdout methods for validation set; for Time Series problems, randomizing is not valid. Validation data should be in order and occur after the training data. With large datasets, you can train on the first 66-75% of the data and validate your model with the remaining 33-25%.

In the case of this problem, which is considered a small dataset, this presents a unique challenge. Consider the below chart of the San Juan Training Data:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/1a6162fd-2d72-4e3e-af7d-286b13eac8db)

I initially tried to use the red line - or anywhere after that point - as a cut off, but my validation data would have neither of the huge outbreaks. I'd therefore consistently get validation loss that was lower than my training loss, because my validation set was never penalized for missing a large outbreak. This made it very difficult to identify overfitting.

If I instead separated the data somewhere in the blue, I was training with a very small amount of data, and was not achieving good results.

The solution is "Walk Forward Validation".

"Walking" through the data like this allows you to simultaneously train the model, and validate it without messing up the order of your data. The downside is typically performance, as this requires refitting the model so many times; however, your choice of the size of x can slow or speed up the process. Below is the training process for the city with the larger history: San Juan. (Iquitos did not have enough training data to fit an LSTM).

My first notable training attempt was simple: A single LSTM layer with 16 cells, and a final fully connected linear layer to convert the layer output to my desired prediction size of one. The blue line represents the walking prediction made after training on all the data prior.

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/b4049135-da73-43ff-b288-e42fb295b340)

My interpretation of this chart was that the network's weights are updating too slowly. I increased the learning rate from 0.01 to 0.1:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/150a15b1-3164-4704-8d92-c396040b9dca)

With the faster learning rate, our network seems to at least identify that the slope should increase around the times of the large outbreaks. I increased the LSTM layer to 64 nodes and retrained:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/fcd4c5b2-29be-472a-919c-db0925e72fe7)

Even better, the predictions increase around the time of the outbreaks, but come down much too slowly. I doubled the LSTM layer size to 128 nodes:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/1f4b8381-5f57-4aa7-a4d4-cf596142b4b3)

Closer, but still a long way from accurately predicting the magnitude of a bad outbreak. Unfortunately, further increases in layer size lead to overfitting and spastic predictions, such as this 256 node network that predicted negative values quite often:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/91a07d93-b9c1-4c79-87f6-bb8ff618bbac)

## LSTM Performance
Of course, there are more quantitative ways than looking at charts to validate models; the consistent prediction window size (x) used in Walk Forward Validation enables the typical loss measurements by simply evaluating the performance across all the validation windows. Interestingly, in this case, the RMSE (Root Mean Squared Error) did not change much throughout all the network architectures I tried, including any of the above that seem to be improving visually. Examining the predictions on the test data showed why.

All the model architectures I tried fell into one of three categories:

1. Models that didn't predict any outbreaks (likely underfit)
2. Models that predicted nearly constant outbreaks (likely overfit)
3. Models that seemed to predict when an outbreak would happen, but continued predicting outbreak numbers for dozens of weeks after the fact. The error from these extended high predictions made these models unusable.

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/e9cb1feb-17c8-4ebb-95c4-3ba82433c8a0)

Ultimately, LSTM's could not perform well for this problem.

## Supervised Learning Models with Lagged Features
Lagging features is a way to use certain Supervised Learning models with time series data. Each feature gets described not only by the feature observations from that time, but by the immediately prior observations as well. This can be done simply with the pandas .shift() method.

This method creates a lot of features, and not all of them will be useful. Therefore, it's best used with models that can penalize or ignore features, such as Lasso Regression or types of decision tree models. I used Random Forest for this project; regression methods with assumptions about normalcy or linearity did not fare well as the distribution of reported cases is an over-dispersed Poisson distribution. I lagged the selected features going back 3 weeks based on the dengue life cycle previously shown above.

## Random Forest Regressor Output
Random Forest Regressors (using cross-validated & grid-searched parameters) created output that passed the eye test for each city.

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/b30fbfef-01f8-407d-8cde-e94d8e4a376c)

These were good for a score of about 24.6 MAE, however, I had one final hypothesis that these models were under-predicting the outbreaks. With these model predictions as inputs, I did a little "post"-processing:

## Exaggerating Outliers in Time Series Data using Basic Calculus
The goal of this final process was to increase the "peaks" in the data significantly, without increasing all the predictions across the board significantly. Here's how to do this simply and automatically:

1. First, predict the data with a model, and take the derivative of the predictions.
2. Then, scale the derivative values by some amount:

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/f294dc82-44ec-4054-a285-a6b2535ffe04)

3. Next, find the integral of the scaled derivative (plain language: convert the line of slopes values back to a line of prediction values):

![image](https://github.com/Monish24/deng-ai-master/assets/54630644/3b870f14-1aa4-44eb-9326-ea98d159648e)

   Note how while the blue Original Prediction starts with a prediction around 10, the orange Integral of the scaled derivative starts from zero; most integral functions will do this by default, since a derivative line says nothing about the scale of the initial values, just their relationship to each other (slope). Therefore, the last step is to add the starting point to ALL the values. In this example, we'd add 10 to every point in the orange line.

   ![image](https://github.com/Monish24/deng-ai-master/assets/54630644/b15c6063-2212-4689-9f9d-6d83c1af1427)

   The result is a set of predictions that increases large values but does not decrease all values!

## Final Performance
The best process I found for this data was to make predictions with Random Forest Regressors with 3 weeks of lagged features, and then to scale the final output as noted above: 60% for San Juan and just 10% for Iquitos. This output scored a MAE of 24.4327, which is currently (887/14215) on DrivenData.org. ( https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/leaderboard/ )
![image](https://github.com/Monish24/deng-ai-master/assets/54630644/2371aa87-6a40-4ce0-8287-d124470790a9)
