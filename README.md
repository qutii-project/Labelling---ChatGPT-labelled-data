# Labelling---ChatGPT-labelled-data

The project aims to create an optimal model for automatically categorizing questions and answers, employing a large dataset of pre-labeled data. Initially, manually prelabelled data was used. Later on, the data was pre labelled using ChatGPT. The system employs a high-level architecture, utilizing comprehensive data preprocessing and an ensemble machine learning approach. System requirements encompass essential Python libraries and dependencies, covering CSV handling, text processing, feature extraction, machine learning frameworks, and visualization, with diverse techniques contributing to improved accuracy and thorough evaluation metrics providing insights into system characteristics. Around 78 percent accuracy was obtained using data from manually pre labelled model. The accuracy is lower for the ChatGPT labelled data. 

# Model_chatgptlabelled _data

This repository contains code for a sequence classification model trained on a dataset of smart city-related questions and answers. The model is based on the BERT (Bidirectional Encoder Representations from Transformers) architecture and is fine-tuned for classifying text into relevant categories such as strategy, science and tech, analysis, factual, taxonomy, management, and ethics and regulation.

The code aims to preprocess the dataset by cleaning the text, augmenting it with synonyms, and encoding it for training. It then loads the pre-trained BERT model and fine-tunes it using the combined training data. After training, the model is evaluated on both the training and testing datasets to assess its performance in terms of accuracy, precision, recall, and F1 score.

Reasons for Performance Issues: The performance of the model fell short of expectations due to several factors. Firstly, the dataset size and complexity might have overwhelmed the resources available in the Google Colab environment, leading to longer execution times and potential crashes. Additionally, fine-tuning a BERT model requires significant computational resources, and the default configurations might not have been optimized for the specific dataset and task at hand. Furthermore, the preprocessing and augmentation techniques applied to the dataset might not have been sufficiently optimized, leading to inefficiencies during training and evaluation.


Edit: 20-03-2024
Classification Approach:
---

1. Bert Classification using Lexical EntitiesApproach:
---


Applied a classification of labels using Feature dimension reduction using Named Entity recognition tag replacement.
Intuition behind the approach being that sentences having their fundamental structure same could be classified as one label.

For eg, 
Question A: What is the purpose of the Canal Istanbul project?
and 
Question B 	What is the purpose of the Marmara link in Istanbul?

Both fall under factual category.

The underlying structure of a sentence are same. Hence both the sentence could be reduced to their minimal form. 
This approach has proven to perform better on text classification as shown  [here](https://www.researchgate.net/publication/370890955_Multi-Class_Document_Classification_Using_Lexical_Ontology-Based_Deep_Learning)



Method used: 

Since Large Language Models such as Bert have huge no of parameters (in millions), they are data hungry, Considering the limited number of training/test data we had, the minority classes were clubbed together to form 5 labels -> analysis, strategy, science_and_tech, ethics_and_regulation, taxonomy.


__Step 1: To perform data preprocessing, 
- Removed unwanted characters.
- Removed any URLs
- Since BERT model handles the cases, str.lower() was not performed
- Replacing the Nouns, Adjectives and Adverbs in the text with their named entities as described in the paper.
- A preprocessed text looks like:

-Before: What is the main strategy of __Smart Cities__ like __Barcelona__ in transforming their city infrastructure and services?

-After: What is the main strategy of __organization__ like __country__ in transforming their city infrastructure and services? 

Below are the Entities used:
('Numerals that do not fall under another type', 'CARDINAL'),\
 ('Absolute or relative dates or periods', 'DATE'),\
 ('Named hurricanes, battles, wars, sports events, etc.', 'EVENT'), \
 ('Buildings, airports, highways, bridges, etc.', 'FAC'), \
 ('Countries, cities, states', 'GPE'), \
 ('Any named language', 'LANGUAGE'), \
 ('Named documents made into laws.', 'LAW'), \
 ('Non-GPE locations, mountain ranges, bodies of water', 'LOC'), \
 ('Monetary values, including unit', 'MONEY'), \
 ('Nationalities or religious or political groups', 'NORP'), \
 ('"first", "second", etc.', 'ORDINAL'), \
 ('Companies, agencies, institutions, etc.', 'ORG'), \
 ('Percentage, including "%"', 'PERCENT'), \
 ('People, including fictional', 'PERSON'), \
 ('Objects, vehicles, foods, etc. (not services)', 'PRODUCT'), \
 ('Measurements, as of weight or distance', 'QUANTITY'), \
 ('Times smaller than a day', 'TIME'), \
 ('Titles of books, songs, etc.', 'WORK_OF_ART'), \
 ('Non-GPE locations, mountain ranges, bodies of water', 'LOC'), \
 ('Miscellaneous entities, e.g. events, nationalities, products or works of art',
  'MISC'), \
 ('Companies, agencies, institutions, etc.', 'ORG'), \
 ('Named person or family.', 'PER') \

##Step 2: Train a classification model using a pretrained Bert model. 

bert-base-uncased model from huggingface was used to finetune on classification task with 5 labels.

Code is available in this [notebook](BERT_classification_with_NER.ipynb).

##Results:

Class: analysis
Accuracy: 303/417

Class: strategy
Accuracy: 256/417

Class: science_and_tech
Accuracy: 186/298

Class: ethics_and_regulation
Accuracy: 38/84

Class: taxonomy
Accuracy: 28/75

Overall accuracy: 61.65% for 5 classes.
A little hyperparameter tuning also resulted in about 66% accuracy with slight changes in resultant confusion matrix.

Validation file with both ground truth and predicted labels can be checked [here](5labelpredictions.csv).

##Challenges and Open items:

-- Google colab provides limited GPU compute per week restricting the number of experiments.

-- Bert model usually requires more than 8000 training pair dataset to produce better results. Hypothesis is that as we gather more data across all labels. 
This model could result in a improved accuracy with little hyperparamte tuning and feature reduction.

-- Robust labelling methods to be used. 

--------------
Edit End: 20-03-2024

