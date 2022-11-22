# Databricks notebook source
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

# COMMAND ----------

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

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
# MAGIC 
# MAGIC #### Business goal 5:
# MAGIC NLP: With the popularity of cryptocurrencies, what are the most important topics or factors people care about regarding cryptocurrencies?
# MAGIC #### Technical proposal： 
# MAGIC Technical proposal: First split the sentences into words, stem, and delete stopwords. Then use CountVectorizer to count the number of words with real meanings. Finally, use TF-IDF to find the words’ importance and extract the most important words. 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 6:
# MAGIC NLP:  What’s people’s attitude on Reddit posts and comments that mention the buzzwords?
# MAGIC #### Technical proposal:
# MAGIC Technical proposal: Reuse the most frequent words acquired earlier, and filter for the sentences that contain the buzzwords. Then perform sentiment analysis on the corpus. Use TextBlob and Vader to find the sentiment polarity. Also, do the data visualization of the sentimental polarity distribution.
# MAGIC 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 7:
# MAGIC NLP: For one of the most popular branches of cryptocurrency, how does sentiment related to dogecoin change with respect to its price?
# MAGIC #### Technical proposal：
# MAGIC Technical proposal: Use NLP to identify posts that mention dogecoins, or look at future forms of dogecoins. Perform sentiment analysis on posts and assign a positive or negative value to each post. To determine the market development potential of dogecoin as an emerging virtual currency.
# MAGIC $$$$

# COMMAND ----------

# MAGIC %md
# MAGIC ## I. Import cleaned data

# COMMAND ----------

df_sub = spark.read.parquet("/FileStore/submissions2")
df_com = spark.read.parquet("/FileStore/comments2")
df_bit = pd.read_csv("/data/csv/Merged_bitcoin.csv")

# COMMAND ----------

df_sub.show(10)

# COMMAND ----------

df_sub.printSchema()

# COMMAND ----------

df_com.show(10)

# COMMAND ----------

df_com.printSchema()

# COMMAND ----------

print(df_bit)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis goals
# MAGIC 
# MAGIC + What is the distribution of text lengths? 
# MAGIC + What are the most common words overall or over time? 
# MAGIC + What are important words according to TF-IDF?

# COMMAND ----------

from pyspark.sql.types import IntegerType

# Write a user-defined function to compute the length of list
slen = f.udf(lambda s: len(s), IntegerType())

# COMMAND ----------

# Word segmentation
df_c = df_com.withColumn('text_split', f.split(col('body'), ' '))
df_s = df_sub.withColumn('title_split', f.split(col('title'), ' '))

# Compute the length of text
df_c = df_c.withColumn('text_len', slen(col('text_split')))
df_s = df_s.withColumn('title_len', slen(col('title_split')))

# COMMAND ----------

# Show the summary of text length
df_c.select('text_len').summary().show()

# COMMAND ----------

# Show the summary of text length
df_s.select('title_len').summary().show()

# COMMAND ----------

# take length data <= 100, which should contain 99% length data to draw the text length graph
df_slen = df_s.select('title_len').filter('title_len <= 100').toPandas()

# COMMAND ----------

# take length data <= 150, which should contain 99% length data to draw the text length graph
df_len = df_c.select('text_len').filter('text_len <= 150').toPandas()

# COMMAND ----------

fig = px.histogram(df_slen.sample(frac=0.05,random_state=1), x='title_len', title="Distribution of Length Of Titles for Submissions", 
                   height=500,template='seaborn',
                  labels={ # replaces default labels by column name
                "title_len": "title length",  "count": "Frequency"
            })
fig.update_layout( # customize font and legend orientation & position
    font_family="Rockwell",
    
)

fig.show()
# fig.write_html('fig_5.html')

# COMMAND ----------

fig = px.histogram(df_len.sample(frac=0.05,random_state=1), x='text_len', title="Distribution of Length Of Texts for Comments", 
                   height=500,template='seaborn',
                  labels={ # replaces default labels by column name
                "text_len": "comment length",  "count": "Frequency"
            })
fig.update_layout( # customize font and legend orientation & position
    font_family="Rockwell",
    
)

fig.show()
# fig.write_html('fig_5.html')

# COMMAND ----------

# MAGIC %md
# MAGIC ## II. Clean the text data using JohnSnowLabs sparkNLP
# MAGIC 
# MAGIC + a. Submission data
# MAGIC + b. Comment data

# COMMAND ----------

# Define words cleaner pipeline for English
eng_stopwords = stopwords.words('english')
# Transform the raw text into the form of document
# for subumission, input should be title
documentAssembler = DocumentAssembler() \
    .setInputCol("body") \ # For Comments
#     .setInputCol("title") \ # For Submission
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

# Word segmentation
tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

# Remove stop words in English
stop_words1 = StopWordsCleaner.pretrained("stopwords_en", "en") \
        .setInputCols(["token"]) \
        .setOutputCol("stop") \
        .setStopWords(eng_stopwords)

# Removes punctuations (keeps alphanumeric chars)
# Convert text to lower-case
normalizer = Normalizer() \
    .setInputCols(["stop"]) \
    .setOutputCol("normalized") \
    .setCleanupPatterns(["""[^\w\d\s]"""])\
    .setLowercase(True)
    
# note that lemmatizer needs a dictionary. So I used the pre-trained
# model (note that it defaults to english)
lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemma')
# # Return hard-stems out of words with the objective of retrieving the meaningful part of the word.
# stemmer = Stemmer() \
#     .setInputCols(["normalized"]) \
#     .setOutputCol("stem")

# Remove stop words in English again
stop_words2 = StopWordsCleaner.pretrained("stopwords_en", "en") \
        .setInputCols(["lemma"]) \
        .setOutputCol("final") \
        .setStopWords(eng_stopwords)

# Finisher converts tokens to human-readable output
finisher = Finisher() \
     .setInputCols(['final']) \
     .setCleanAnnotations(False)

# Set up the pipeline
pipeline = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    tokenizer,
    stop_words1,
    normalizer,
    lemmatizer,
    stop_words2,
    finisher
])

# COMMAND ----------

# MAGIC %md
# MAGIC #### a. Submission data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### a.1 Implement pipline

# COMMAND ----------

df_text_s = df_sub.select("title")
# Fit the dataset into the pipeline
df_cleaned_s = pipeline.fit(df_text_s).transform(df_text_s)

# COMMAND ----------

# Save the cleaned data to DBFS
df_cleaned_s.write.parquet("/FileStore/pipelinedSubmission")

# COMMAND ----------

df_cleaned_s.printSchema()

# COMMAND ----------

# Check the cleaned text
df_cleaned_s.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### a.2 Compute word count to find the most common words

# COMMAND ----------

# Transform the dataframe into rdd
text_rdd_s = df_cleaned_s.select('finished_final').rdd

# COMMAND ----------

# Map the rdd by assigning all words with 1 count
text_s = text_rdd_s.map(list).map(lambda x: x[0])
text_s = text_s.flatMap(lambda x:x).map(lambda x: (x,1))

# COMMAND ----------

# Reduce the rdd by aggregate the same word
text_count_s = text_s.reduceByKey(lambda x,y:(x+y)).sortBy(lambda x:x[1], ascending=False)
# Take the top 50 words with their counts
text_count_s.take(50)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### a.3 Compute the Tf-idf to find the most important words

# COMMAND ----------

# Initial a CountVectorizer
cv = CountVectorizer(inputCol="finished_final", 
                     outputCol="tf",
                     vocabSize=1000) # consider only the 1000 most frequent terms

# Fit the cleaned data
cv_model_s = cv.fit(df_cleaned_s)
df_cv_s = cv_model.transform(df_cleaned_s)

# COMMAND ----------

# Initial a TfidfVectorizer based on the result of CountVectorizer
idf = IDF(inputCol='tf', 
        outputCol='tfidf',
         )

# Fit the data
idf_model_s = idf.fit(df_cv_s)
df_idf_s = idf_model.transform(df_cv_s)

# COMMAND ----------

vocab_s = spark.createDataFrame(pd.DataFrame({'word': cv_model_s.vocabulary, 
                                            'tfidf': idf_model_s.idf}))

# COMMAND ----------

vocab_s = vocab_s.sort('tfidf', ascending=False)
vocab_s.show(50)

# COMMAND ----------

# MAGIC %md
# MAGIC #### b. Comment data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### b.1 Implement pipline

# COMMAND ----------

df_text_c = df_com.select("body")
# Fit the dataset into the pipeline
df_cleaned_c = pipeline.fit(df_text_c).transform(df_text_c)

# COMMAND ----------

# Save the cleaned dataframe to DBFS
df_cleaned_c.write.parquet("/FileStore/pipelinedComment")

# COMMAND ----------

df_cleaned_c.printSchema()

# COMMAND ----------

# Check the cleaned text
df_cleaned_c.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### b.2 Compute word count to find the most common words

# COMMAND ----------

# Transform the dataframe into rdd
text_rdd = df_cleaned_c.select('finished_final').rdd

# COMMAND ----------

# Map the rdd by assigning all words with 1 count
text = text_rdd.map(list).map(lambda x: x[0])
text = text.flatMap(lambda x:x).map(lambda x: (x,1))

# COMMAND ----------

# Reduce the rdd by aggregate the same word
text_count = text.reduceByKey(lambda x,y:(x+y)).sortBy(lambda x:x[1], ascending=False)
# Take the top 50 words with their counts
text_count.take(50)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### b.3 Compute the Tf-idf to find the most important words

# COMMAND ----------

# Initial a CountVectorizer
cv = CountVectorizer(inputCol="finished_final", 
                     outputCol="tf",
                     vocabSize=1000) # consider only the 1000 most frequent terms

# Fit the cleaned data
cv_model = cv.fit(df_cleaned_c)
df_cv = cv_model.transform(df_cleaned_c)

# COMMAND ----------

# Initial a TfidfVectorizer based on the result of CountVectorizer
idf = IDF(inputCol='tf', 
         outputCol='tfidf')

# Fit the data
idf_model = idf.fit(df_cv)
df_idf = idf_model.transform(df_cv)

# COMMAND ----------

vocab_c = spark.createDataFrame(pd.DataFrame({'word': cv_model.vocabulary, 
                                            'tfidf': idf_model.idf}))

# COMMAND ----------

vocab_c = vocab_c.sort('tfidf', ascending=False)
vocab_c.show(50)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## III. Sentiment Model

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Identify important keywords for your reddit data and use regex searches, create new variables

# COMMAND ----------

# Use Regex to find submissions that mention different cryptocurrencies


df = df_sub

df = df.withColumn('BTC', when(df.title.rlike('(?i)bitcoin|(?i)btc'), 1).otherwise(0))
df = df.withColumn('ETH', when(df.title.rlike('(?i)etherium|ETH'), 1).otherwise(0))
df = df.withColumn('USDT', when(df.title.rlike('(?i)USDT|(?i)tether'), 1).otherwise(0))
df = df.withColumn('USDC', when(df.title.rlike('(?i)USDC'), 1).otherwise(0))
df = df.withColumn('BNB', when(df.title.rlike('(?i)BNB'), 1).otherwise(0))
df = df.withColumn('XRP', when(df.title.rlike('(?i)XRP'), 1).otherwise(0))
df = df.withColumn('BUSD', when(df.title.rlike('(?i)BUSD|(?i)Binance USD'), 1).otherwise(0))
df = df.withColumn('ADA', when(df.title.rlike('(?i)cardano|(?i)ADA'), 1).otherwise(0))
df = df.withColumn('SOL', when(df.title.rlike('(?i)solana|(?i)SOL'), 1).otherwise(0))
df = df.withColumn('DOG', when(df.title.rlike('(?i)dogecoin|(?i)DOGE'), 1).otherwise(0))

df_sml = df.select("BTC","ETH","USDT",'USDC','BNB','XRP','BUSD','ADA','SOL','DOG')

res = df_sml.groupBy().sum().collect()
