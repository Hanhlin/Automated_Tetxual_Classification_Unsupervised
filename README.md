# Automated_Tetxual_Classification_Unsupervised
Perform descriptive modeling and clustering of textual data

## Instructions
Topics per cluster folder and confusion matrix are generated to current directory.

Two plots are generated and pop up while executing the program.

## Sources
- ClusterEvaluation.java
- Kmeans.java
- KmeansPP.java
- Main.java
- preProcess.java
- SimilarityMeasure.java
- SVD.java
- Visualizer.java

## Dependency Files
- stopwords.txt
- train data folder
- test data folder

## Requirements
- StanfordCoreNLP
- Jama
- gral-core

## Process 
First, define all paths for training data, testing data, stopwords, and output data.
Then read and preprocess training data to generate tf-idf.
Kmeans model is trained on given document data, and use this model to classify test data.
Generate tf-idf for test data using tf from test data and idf from train data.
Use Knn to classify clusters for test data using clusters of the trained model.
