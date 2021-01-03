# Resturant-Review-NLP
RESTAURANT REVIEWS PREDICTION (NATURAL LANGUAGE PROCESSING)
The purpose of this analysis is to build a prediction model to predict whether a review of the restaurant is positive or negative.

Dataset: Restaurant_Reviews.tsv is a dataset from Kaggle datasets which consists of 1000 reviews on a restaurant.

To build a model to predict if the review is positive or negative, the following steps are performed.

1- Importing Dataset 
2- Preprocessing Dataset
3- Vectorization 
4- Training and Classification 
5- Analysis Conclusion

1)Importing Dataset Importing the Restaurant Review dataset using pandas library.

2) Preprocessing Dataset 
     Each review undergoes a preprocessing step, where all the vague information is removed. 
         1- Removing the Stopwords, numeric and special characters.
         2- Normalizing each review using the approach of stemming.

3) Vectorization 
     From the cleaned dataset, potential features are extracted and are converted to the numerical format. The vectorization techniques are used to convert textual data to a     numerical format. Using vectorization, a matrix is created where each column represents a feature and each row represents an individual review

4) Training and Classification 
     Further the data is splitted into training and testing set using Cross Validation technique. This data is used as input to classification algorithm.

5) Analysis and Conclusion
      In this study, an attempt has been made to classify sentiment analysis for restaurant reviews using machine learning techniques.
      Evaluation metrics used here are accuracy, precision and recall.
