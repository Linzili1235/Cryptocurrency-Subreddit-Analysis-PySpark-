# Databricks notebook source
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression, LinearSVC,DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml import Pipeline, Model
from pyspark.ml.pipeline import PipelineModel
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

# COMMAND ----------

#import library
import pyspark.sql.functions as f
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from operator import add
from functools import reduce
from pyspark.ml.feature import CountVectorizer,IDF
from pyspark.sql.functions import concat_ws

import string
import nltk
from nltk.corpus import stopwords
from itertools import chain

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import roc_curve, roc_auc_score

# COMMAND ----------

spark = SparkSession.builder \
        .appName("RedditNLP") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.4.2") \
    .master('yarn') \
    .getOrCreate()

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md
# MAGIC #### Business goal 8:
# MAGIC ML: We want to predict the sub-community a comment in the cryptocurrency subreddit belongs to. 
# MAGIC #### Technical proposal:
# MAGIC Technical proposal: Based on the user's theme, we recommend what subreddit theme is suitable for him. Build a Naive Bayes model to determine the user's topic, and finally, use the model to suggest his own suitable topic for the new post.
# MAGIC 
# MAGIC !!!! Using word embedding to vectorize the comments. Implement RNN and LSTM to predict the label and compare their performances in terms of prediction accuracy. 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 9:
# MAGIC ML: We want to help Reddit to find out whether a cryptocurrency’s comment is controversial or not. 
# MAGIC #### Technical proposal:
# MAGIC First, properly choose important and distinct variables that can be used to predict controversial variables(dummy variable). And then utilize regression and clustering algorithms to predict the values of controversial variables. Finally, compare the real variables and predicted variables and get the algorithm prediction accuracy and find out whether the algorithm can be put into real-world implementation. 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 10:
# MAGIC ML: We want to help Reddit to judge a cryptocurrency’s comment’s quality based on the overall ratings it received.
# MAGIC #### Technical proposal:
# MAGIC First, properly choose important and distinct variables that can be used to predict the score variable(integer variable). And then utilize clustering algorithms to predict the values of the score variable. Finally, compare the real variables and predicted variables and get the algorithm prediction accuracy and find out whether the algorithm can be put into real-world implementation.
# MAGIC $$$$

# COMMAND ----------

colN = df_com.count()

# COMMAND ----------

colN

# COMMAND ----------

# Read in data
df_sub = spark.read.parquet("/FileStore/submissions2")
df_com = spark.read.parquet("/FileStore/sentimentcomment")
df_bit = pd.read_csv("../../data/csv/Merged_bitcoin.csv", index_col=False)

# COMMAND ----------

df_com.printSchema()

# COMMAND ----------

df_com.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC # Q8

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data collection

# COMMAND ----------

df_com_ml = df_com.withColumn('predict_label', when(df_com.text.rlike('(?i)pow'), 'pow')
                                   .when(df_com.text.rlike('(?i)fork'), 'fork')
                                   .when(df_com.text.rlike('(?i)bitcoin|(?i)btc'), 'btc')
                                   .when(df_com.text.rlike('(?i)ethereum|ETH'), 'eth')
                                   .when(df_com.text.rlike('(?i)polygon'), 'polygon')
                                   .when(df_com.text.rlike('(?i)algorand'),'algorand')
                                   .when(df_com.text.rlike('(?i)robinhood'),'robinhood')
                                   .otherwise('N/A'))

# COMMAND ----------

df_com_ml.show(5)

# COMMAND ----------

df_com_ml.groupBy("predict_label").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data cleaning and preparation

# COMMAND ----------

# Remove N/As from "predict_label" column
df_com_ml = df_com_ml.filter(col('predict_label') != 'N/A')

# COMMAND ----------

# Select "body" and "predict_label" columns to form the dataframe for ML
df_ml = df_com_ml.select("body", "predict_label")

# COMMAND ----------

df_ml1 = df_com_ml.select("text", "predict_label")

# COMMAND ----------

df_ml.show(5)

# COMMAND ----------

# Collect labels
txt_label = df_ml.select("predict_label").distinct().toPandas()
txt_label.head(10)

# COMMAND ----------

ls_label = list(txt_label["predict_label"])
ls_label

# COMMAND ----------

# Define words cleaner pipeline for English
eng_stopwords = stopwords.words('english')
# Transform the raw text into the form of document
# for subumission, input should be title
documentAssembler = DocumentAssembler() \
    .setInputCol("body") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

# # Word segmentation
# tokenizer = Tokenizer() \
#     .setInputCols(["sentence"]) \
#     .setOutputCol("token")

# Remove stop words in English
# stop_words1 = StopWordsCleaner.pretrained("stopwords_en", "en") \
#         .setInputCols(["sentence"]) \
#         .setOutputCol("stop") \
#         .setStopWords(eng_stopwords)

# Removes punctuations (keeps alphanumeric chars)
# Convert text to lower-case
# normalizer = Normalizer() \
#     .setInputCols(["stop"]) \
#     .setOutputCol("normalized") \
#     .setCleanupPatterns(["""[^\w\d\s]"""])\
#     .setLowercase(True)
    
# note that lemmatizer needs a dictionary. So I used the pre-trained
# model (note that it defaults to english)
# lemmatizer = LemmatizerModel.pretrained() \
#      .setInputCols(['normalized']) \
#      .setOutputCol('lemma')
# # Return hard-stems out of words with the objective of retrieving the meaningful part of the word.
# stemmer = Stemmer() \
#     .setInputCols(["normalized"]) \
#     .setOutputCol("stem")

# Remove stop words in English again
# stop_words2 = StopWordsCleaner.pretrained("stopwords_en", "en") \
#         .setInputCols(["lemma"]) \
#         .setOutputCol("final") \
#         .setStopWords(eng_stopwords)

# Finisher converts tokens to human-readable output
finisher = Finisher() \
     .setInputCols(['sentence']) \
     .setOutputCols(['sentences_string'])\
     .setOutputAsArray(True)\
     .setCleanAnnotations(False)

# Set up the pipeline
pipeline = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
#     tokenizer,
#     stop_words1,
#     normalizer,
#     lemmatizer,
#     stop_words2,
    finisher
])

# COMMAND ----------

df_cleaned_txt = pipeline.fit(df_ml).transform(df_ml)

# COMMAND ----------

df_cleaned_txt.first()['sentences_string']

# COMMAND ----------

# Build the corpus
new_corpus = []
for item in ls_label:
    corpus.append(collectTxt(item))
print(len(corpus))

# COMMAND ----------

# Check raw corpus
corpus[0][:5]

# COMMAND ----------

# Cleaning
stop_words = stopwords.words("english")

def cleanText(inp):
    inp = inp.lower().strip() # convert to lower-case
    inp = ' '.join([word for word in inp.split(' ') if word not in stop_words]) # remove stopwords
    inp = re.sub("[^a-zA-Z]"," ", inp)
    inp = inp.encode('ascii', 'ignore').decode()
    return inp

def cleanApply(text):
    for i, sent in enumerate(text):
        text[i] = cleanText(sent)

# COMMAND ----------

for text in corpus:
    cleanApply(text)

# COMMAND ----------

# Check cleaned corpus
corpus[0][:5]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split data into train, test, and split

# COMMAND ----------

# convert label to index
labelEncoder = StringIndexer(inputCol='predict_label',outputCol='label').fit(df_ml1)

# COMMAND ----------

labelEncoder.transform(df_ml1).show(5)

# COMMAND ----------

df_ml1 = labelEncoder.transform(df_ml1)

# COMMAND ----------

train_data, test_data, predict_data = df_ml1.randomSplit([0.8, 0.18, 0.02], 24)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define the stage of pipline

# COMMAND ----------

tokenizer = Tokenizer(inputCol='text',outputCol='mytokens')
stopwords_remover = StopWordsRemover(inputCol='mytokens',outputCol='filtered_tokens')
vectorizer = CountVectorizer(inputCol='filtered_tokens',outputCol='rawFeatures')
idf = IDF(inputCol='rawFeatures',outputCol='vectorizedFeatures')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 1: Logistic Regression

# COMMAND ----------

lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')

# COMMAND ----------

pipeline_lr = Pipeline(stages=[tokenizer,
                               stopwords_remover,
                               vectorizer,
                               idf,
                               lr])

# COMMAND ----------

model_rf2 = pipeline_lr.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Model Test Results for Logistic Regression

# COMMAND ----------

predictions1 = model_rf2.transform(test_data)
#ACC
evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions1)

# MSE
mse_lr = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                    metricName="mse").evaluate(predictions1)

# VAR
var_lr = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                   metricName="var").evaluate(predictions1)

print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))
print("MSE = %g" % mse_lr)
print("Var = %g" % var_lr)

# COMMAND ----------



# COMMAND ----------

y_pred=predictions1.select("prediction").collect()
y_orig=predictions1.select("label").collect()
cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

f,ax=plt.subplots()
sns.heatmap(cm,annot=True,ax=ax)

ax.set_title('confusion matrix') 
ax.set_xlabel('Prediction')
ax.set_ylabel('Label')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 2&3: DecisionTree with maxDepth=3 & maxDepth=5

# COMMAND ----------

dt1 = DecisionTreeClassifier(labelCol="label", featuresCol="vectorizedFeatures", maxDepth =3)

# COMMAND ----------

pipeline_dt1 = Pipeline(stages=[tokenizer,
                               stopwords_remover,
                               vectorizer,
                               idf,
                               dt1])

# COMMAND ----------

model_dt1 = pipeline_dt1.fit(train_data)

# COMMAND ----------

predictions = model_dt1.transform(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Model Test Results for Decision Tree 1

# COMMAND ----------

predictions2 = model_dt1.transform(test_data)
#ACC
evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions2)

# MSE
mse_dt = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                    metricName="mse").evaluate(predictions2)

# VAR
var_dt = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                   metricName="var").evaluate(predictions2)

print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))
print("MSE = %g" % mse_lr)
print("Var = %g" % var_lr)

# COMMAND ----------

y_pred=predictions2.select("prediction").collect()
y_orig=predictions2.select("label").collect()
cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

f,ax=plt.subplots()
sns.heatmap(cm,annot=True,ax=ax)

ax.set_title('confusion matrix') 
ax.set_xlabel('Prediction')
ax.set_ylabel('Label')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 3 with different hyperparameter 

# COMMAND ----------

dt2 = DecisionTreeClassifier(labelCol="label", featuresCol="vectorizedFeatures", maxDepth =5)
pipeline_dt2 = Pipeline(stages=[tokenizer,
                               stopwords_remover,
                               vectorizer,
                               idf,
                               dt2])

# COMMAND ----------

model_dt2 = pipeline_dt2.fit(train_data)

# COMMAND ----------

predictions = model_dt2.transform(test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Model Test Results for Decision Tree 2

# COMMAND ----------

predictions3 = model_dt2.transform(test_data)
#ACC
evaluatorRF = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions3)

# MSE
mse_lr = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                    metricName="mse").evaluate(predictions3)

# VAR
var_lr = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                   metricName="var").evaluate(predictions3)

print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))
print("MSE = %g" % mse_lr)
print("Var = %g" % var_lr)

# COMMAND ----------

y_pred=predictions3.select("prediction").collect()
y_orig=predictions3.select("label").collect()
cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

f,ax=plt.subplots()
sns.heatmap(cm,annot=True,ax=ax)

ax.set_title('confusion matrix') 
ax.set_xlabel('Prediction')
ax.set_ylabel('Label')

# COMMAND ----------

dic1 = {'Model':['logistic regression','decision tree 1','decision tree 2'],
       'Accuracy':[0.988981,0.893128,0.915337],
       'MSE':[0.0503525,0.0503525,0.400393],
       'VAR':[0.953362,0.953362,0.604116]}

# COMMAND ----------

new = pd.DataFrame.from_dict(dic1)
  
new
