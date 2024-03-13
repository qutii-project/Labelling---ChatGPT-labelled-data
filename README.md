# Labelling---ChatGPT-labelled-data

The project aims to create an optimal model for automatically categorizing questions and answers, employing a large dataset of pre-labeled data. Initially, manually prelabelled data was used. Later on, the data was pre labelled using ChatGPT. The system employs a high-level architecture, utilizing comprehensive data preprocessing and an ensemble machine learning approach. System requirements encompass essential Python libraries and dependencies, covering CSV handling, text processing, feature extraction, machine learning frameworks, and visualization, with diverse techniques contributing to improved accuracy and thorough evaluation metrics providing insights into system characteristics. Around 78 percent accuracy was obtained using data from manually pre labelled model. The accuracy is lower for the ChatGPT labelled data. 

# Model_chatgptlabelled _data

This repository contains code for a sequence classification model trained on a dataset of smart city-related questions and answers. The model is based on the BERT (Bidirectional Encoder Representations from Transformers) architecture and is fine-tuned for classifying text into relevant categories such as strategy, science and tech, analysis, factual, taxonomy, management, and ethics and regulation.

The code aims to preprocess the dataset by cleaning the text, augmenting it with synonyms, and encoding it for training. It then loads the pre-trained BERT model and fine-tunes it using the combined training data. After training, the model is evaluated on both the training and testing datasets to assess its performance in terms of accuracy, precision, recall, and F1 score.

Reasons for Performance Issues: The performance of the model fell short of expectations due to several factors. Firstly, the dataset size and complexity might have overwhelmed the resources available in the Google Colab environment, leading to longer execution times and potential crashes. Additionally, fine-tuning a BERT model requires significant computational resources, and the default configurations might not have been optimized for the specific dataset and task at hand. Furthermore, the preprocessing and augmentation techniques applied to the dataset might not have been sufficiently optimized, leading to inefficiencies during training and evaluation.
