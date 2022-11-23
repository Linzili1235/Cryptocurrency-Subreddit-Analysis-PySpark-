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
from wordcloud import WordCloud

# COMMAND ----------

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go


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
# MAGIC From the wordcloud about most common words in the submission topics, we can see that users who post the Reddit submissions under cryptocurrency have great concerns about topics such as "bitcoin","cryptocurrencies","eth" and so on. Although with the popularity of different cryptocurrencies in recent years, such as dogecoin, ethereum, bitcoin still attracts extensive attention from the public. From the most common words shown above, the main purpose of users posting the submissions is to buy, invest, and sell cryptocurrencies since words like "price","market", "wallet" are of great amount in the topics. 
# MAGIC 
# MAGIC From the wordcloud about most important words of the comment body, we can find that users who comment under the topic of cryptocurrency are likely to bring up words such as "pow","fork","pop" and so on. Different from the users who come up with the question about the prices of cryptocurrencies in the submissions, commenters tend to give replies in a more professional way, such as discussing about different protocols and crypto terminalogies like pow,algorand and polygon. Just as the "art" shown in the wordcloud, in about 20 words (see from the distribution of the comments from cryptocurrency, the majority of comment length lies below 20), commenters describe cryptocurrencies as arts while the question posters see them as products with values.
# MAGIC 
# MAGIC #### Business goal 6:
# MAGIC NLP:  What’s people’s attitude on Reddit posts and comments that mention the buzzwords?
# MAGIC #### Technical proposal:
# MAGIC Technical proposal: Reuse the most frequent words acquired earlier, and filter for the sentences that contain the buzzwords. Then perform sentiment analysis on the corpus. Use TextBlob and Vader to find the sentiment polarity. Also, do the data visualization of the sentimental polarity distribution.
# MAGIC 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 7:
# MAGIC NLP: For one of the most popular branches of cryptocurrency, how does sentiment related to bitcoin change with respect to its price?
# MAGIC #### Technical proposal：
# MAGIC Technical proposal: Use NLP to identify posts that mention bitcoins, or look at future forms of bitcoins. Perform sentiment analysis on posts and assign a positive or negative value to each post. To determine the market development potential of bitcoin as an emerging virtual currency.
# MAGIC $$$$

# COMMAND ----------

# Import data
df_sub_p = spark.read.parquet("/FileStore/pipelinedSubmission")
df_com_p = spark.read.parquet("/FileStore/pipelinedComment")

df_sub = spark.read.parquet("/FileStore/submissions2")
df_com = spark.read.parquet("/FileStore/comments2")
df_bit = pd.read_csv("../../data/csv/Merged_bitcoin.csv", index_col=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## I. Insights on External data 

# COMMAND ----------

df_bit = df_bit.drop(columns=['Unnamed: 0'])
df_bit.head()

# COMMAND ----------

# Show statistics of the data
df_bit['Count'].describe()

# COMMAND ----------

df_bit['Bitcoin_Price'].describe()

# COMMAND ----------

# Distribution of Bitcoin Price
fig = px.histogram(df_bit, x='Bitcoin_Price', title="Distribution of Bitcoin Price", 
                   height=500,template='seaborn',
                  labels={ # replaces default labels by column name
                "Bitcoin_Price": "Bitcoin Price($)",  "count": "Frequency"
            })
fig.update_layout( # customize font and legend orientation & position
    font_family="Rockwell",
    
)

fig.show()
fig.write_html("../../images/figure_distri_BTC.html")

# COMMAND ----------

# Price of Bitcoin over time  in S

# Sort the Date because there some inconsistent dates
data = df_bit.sort_values(by=['Date'])

fig = px.line(data,x = "Date", y= 'Bitcoin_Price', title = 'Price of Bitcoin($) Over Time',template='seaborn' )

# Add range slider
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()
fig.write_html("../../images/figure_line.html")

# COMMAND ----------

# MAGIC %md
# MAGIC ## II. Natural Language Processing

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analysis goals
# MAGIC 
# MAGIC + What is the distribution of text lengths? 
# MAGIC + What are the most common words overall or over time? 
# MAGIC + What are important words according to TF-IDF?

# COMMAND ----------

df_sub.show(10)

# COMMAND ----------

df_sub.printSchema()

# COMMAND ----------

df_com.show(10)

# COMMAND ----------

df_com.printSchema()

# COMMAND ----------

df_bit.head()

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
fig.write_html('../../images/fig_distr_Submission.html')

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
fig.write_html('../../images/fig_distr_Comment.html')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Clean the text data using JohnSnowLabs sparkNLP
# MAGIC 
# MAGIC + a. Submission data
# MAGIC + b. Comment data

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

df_text_s = df_sub
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
# text_rdd_s = df_cleaned_s.select('finished_final').rdd
text_rdd_s = df_sub_p.select('finished_final').rdd

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
df_cv_s = cv_model_s.transform(df_cleaned_s)

# COMMAND ----------

df_cv_s.show(10)

# COMMAND ----------

## df_cv_s1 = df_cv_s.withColumn('tf_dense',df_cv_s.select('tf').map(lambda x: Vectors.sparse(x)))

# COMMAND ----------

# Initial a TfidfVectorizer based on the result of CountVectorizer
idf = IDF(inputCol='tf', 
        outputCol='tfidf',
         )

# Fit the data
idf_model_s = idf.fit(df_cv_s)
df_idf_s = idf_model_s.transform(df_cv_s)

# COMMAND ----------

vocab_s = spark.createDataFrame(pd.DataFrame({'word': cv_model_s.vocabulary, 
                                            'tfidf': idf_model_s.idf}))

# COMMAND ----------

vocab_s = vocab_s.sort('tfidf', ascending=False)

# COMMAND ----------

tf_df_s = text_count_s.collect()
tf_df_s = pd.DataFrame(tf_df_s)
print(len(tf_df_s))

# COMMAND ----------

tf_df_s.head()

# COMMAND ----------

# Normalize TF-IDF
maxX = tf_df_s.max()[1]
minX = tf_df_s.min()[1]

norm_tfdf = tf_df_s[:]
norm_tfdf[1] = (norm_tfdf[1]-minX)/(maxX-minX)

print(norm_tfdf.head())

# COMMAND ----------

# Construct the wordcloud for Submission based on word count
x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

tmpDict = pd.Series(norm_tfdf[1].values,index=norm_tfdf[0]).to_dict()
Cloud = WordCloud(background_color="white", max_words=50,mask=mask).generate_from_frequencies(tmpDict)
plt.figure(figsize=[10,10])
plt.imshow(Cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC From the wordcloud about most common words in the submission topics, we can see that users who post the Reddit submissions under cryptocurrency have great concerns about topics such as "bitcoin","cryptocurrencies","eth" and so on. Although with the popularity of different cryptocurrencies in recent years, such as dogecoin, ethereum, bitcoin still attracts extensive attention from the public. From the most common words shown above, the main purpose of users posting the submissions is to buy, invest, and sell cryptocurrencies since words like "price","market", "wallet" are of great amount in the topics. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### b. Comment data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### b.1 Implement pipline

# COMMAND ----------

df_text_c = df_com
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
cv_model_c = cv.fit(df_cleaned_c)
df_cv = cv_model_c.transform(df_cleaned_c)

# COMMAND ----------

# Initial a TfidfVectorizer based on the result of CountVectorizer
idf = IDF(inputCol='tf', 
         outputCol='tfidf')

# Fit the data
idf_model = idf.fit(df_cv)
df_idf = idf_model.transform(df_cv)

# COMMAND ----------

vocab_c = spark.createDataFrame(pd.DataFrame({'word': cv_model_c.vocabulary, 
                                            'tfidf': idf_model.idf}))

# COMMAND ----------

vocab_c = vocab_c.sort('tfidf', ascending=False)


# COMMAND ----------

tf_df_c = vocab_c.collect()
tf_df_c = pd.DataFrame(tf_df_c)
print(len(tf_df_c))

# COMMAND ----------

# Normalize TF-IDF
maxX = tf_df_c.max()[1]
minX = tf_df_c.min()[1]

norm_tfdf = tf_df_c[:]
norm_tfdf[1] = (norm_tfdf[1]-minX)/(maxX-minX)

print(norm_tfdf.head())

# COMMAND ----------

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

tmpDict = pd.Series(norm_tfdf[1].values,index=norm_tfdf[0]).to_dict()
Cloud = WordCloud(background_color="white", max_words=50,mask=mask).generate_from_frequencies(tmpDict)
plt.figure(figsize=[10,10])
plt.imshow(Cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC From the wordcloud about most important words of the comment body, we can find that users who comment under the topic of cryptocurrency are likely to bring up words such as "pow","fork","pop" and so on. Different from the users who come up with the question about the prices of cryptocurrencies in the submissions, commenters tend to give replies in a more professional way, such as discussing about different protocols and crypto terminalogies like pow,algorand and polygon. Just as the "art" shown in the wordcloud, in about 20 words (see from the distribution of the comments from cryptocurrency, the majority of comment length lies below 20), commenters describe cryptocurrencies as arts while the question posters see them as products with values.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Identify important keywords for your reddit data and use regex searches to create at least two dummy variables

# COMMAND ----------

# Define a function to union the words into a sentence
wordToText = f.udf(lambda x:' '.join(x))
# Apply the function to create the cleaned text
df_senti_c = df_cleaned_c.withColumn('text', wordToText(f.col('finished_final')))

# COMMAND ----------

# Use Regex to find comments that mention different cryptocurrencies

df = df_senti_c

df = df.withColumn('crypto_term', when(df.text.rlike('(?i)pow'), 'pow')
                                   .when(df.text.rlike('(?i)fork'), 'fork')
                                   .when(df.text.rlike('(?i)cryptocurrenc'), 'cryptocurrencies')
                                   .otherwise('N/A'))
df = df.withColumn('BTC', when(df.text.rlike('(?i)bitcoin|(?i)btc'), 1).otherwise(0))
df = df.withColumn('ETH', when(df.text.rlike('(?i)ethereum|ETH'), 1).otherwise(0))
df = df.withColumn('polygon', when(df.text.rlike('(?i)polygon'), 1).otherwise(0))
df = df.withColumn('algorand', when(df.text.rlike('(?i)algorand'), 1).otherwise(0))
df = df.withColumn('robinhood', when(df.text.rlike('(?i)robinhood'), 1).otherwise(0))

# COMMAND ----------

df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## III. Sentiment Model

# COMMAND ----------

# Define sentiment pipeline
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained('tfhub_use', lang="en") \
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

sentimentdl = SentimentDLModel().pretrained('sentimentdl_use_twitter')\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
    stages = [
        documentAssembler,
        use,
        sentimentdl
    ])

# COMMAND ----------

df_senti_c = df
# Run the pipeline
pipelineModel = nlpPipeline.fit(df_senti_c)

result = pipelineModel.transform(df_senti_c)

# Get the sentiment
result = result.withColumn('labels', result.sentiment.result[0]) 

# COMMAND ----------

# Show some of the results
result.select('labels').show(5)

# COMMAND ----------

result.printSchema()

# COMMAND ----------

# delect unnecessary columns
result = result.drop('document', 'token','sentence','normalized', 'stop','lemma','stem', 'final', 'tf', 'tfidf', 'sentence_embeddings', 'sentiment')

# COMMAND ----------

result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary statistics and visualizations

# COMMAND ----------

# Basic summary
result.groupby('labels').count().show()

# COMMAND ----------

# Filter the rows with null labels
result = result.filter(result.labels.isNotNull())

# COMMAND ----------

# Two way tables
result.groupby('labels','crypto_term').count().show()

# COMMAND ----------

result.printSchema()

# COMMAND ----------


# df = df.withColumn('crypto_term', when(df.text.rlike('(?i)pow'), 'pow')
#                                    .when(df.text.rlike('(?i)fork'), 'fork')
#                                    .when(df.text.rlike('(?i)cryptocurrenc'), 'cryptocurrencies')
#                                    .otherwise('N/A'))
# df = df.withColumn('BTC', when(df.text.rlike('(?i)bitcoin|(?i)btc'), 1).otherwise(0))
# df = df.withColumn('ETH', when(df.text.rlike('(?i)etherium|ETH'), 1).otherwise(0))
# df = df.withColumn('polygon', when(df.text.rlike('(?i)polygon'), 1).otherwise(0))
# df = df.withColumn('algorand', when(df.text.rlike('(?i)algorand'), 1).otherwise(0))
# df = df.withColumn('robinhood', when(df.text.rlike('(?i)robinhood'), 1).otherwise(0))

# COMMAND ----------

df_cryptoTerm = result.filter(col('crypto_term') != 'N/A')
df_pow = result.filter(col('crypto_term') == 'pow')
df_fork = result.filter(col('crypto_term') == 'fork')
df_cryptocurrencies = result.filter(col('crypto_term') == 'cryptocurrencies')
df_BTC = result.filter(col('BTC') == 1)
df_polygon = result.filter(col('polygon') == 1)
df_algorand = result.filter(col('algorand') == 1)
df_robinhood = result.filter(col('robinhood') == 1)
df_ETH = result.filter(col('ETH') == 1)

# COMMAND ----------

# Summary Tables
cryptoTerm = df_cryptoTerm.groupby('labels').count().sort('labels').toPandas()
pow = df_pow.groupby('labels').count().sort('labels').toPandas()
fork = df_fork.groupby('labels').count().sort('labels').toPandas()
crypto = df_cryptocurrencies.groupby('labels').count().sort('labels').toPandas()


# COMMAND ----------

# Summary Tables
BTC = df_BTC.groupby('labels').count().sort('labels').toPandas()

# COMMAND ----------

polygon = df_polygon.groupby('labels').count().sort('labels').toPandas()

# COMMAND ----------

algorand = df_algorand.groupby('labels').count().sort('labels').toPandas()

# COMMAND ----------

# robin = df_robinhood.groupby('labels').count().sort('labels').toPandas()

# COMMAND ----------

ETH = df_ETH.groupby('labels').count().sort('labels').toPandas()

# COMMAND ----------

cryptoTerm.to_csv('../../data/csv/cryptoTerm.csv')
pow.to_csv('../../data/csv/pow.csv')
fork.to_csv('../../data/csv/fork.csv')
crypto.to_csv('../../data/csv/crypto.csv')
BTC.to_csv('../../data/csv/BTC.csv')
polygon.to_csv('../../data/csv/polygon.csv')
algorand.to_csv('../../data/csv/algorand.csv')
#robin.to_csv('../../data/csv/robin.csv')
ETH.to_csv('../../data/csv/ETH.csv')

# COMMAND ----------

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])

labels = ['negative', 'neutral', 'positive']

fig.add_trace(go.Pie(labels=labels, values=cryptoTerm['count'], marker_colors=['#3D65A5', '#3DA57D', '#F05039'], name="Crypto Terms"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=pow['count'], marker_colors=['#3D65A5', '#3DA57D', '#F05039'], name="POW"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=fork['count'], marker_colors=['#3D65A5', '#3DA57D', '#F05039'], name="Fork"),
              2, 1)
fig.add_trace(go.Pie(labels=labels, values=crypto['count'], marker_colors=['#3D65A5', '#3DA57D', '#F05039'], name="Cryptocurrencies"),
              2, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    title = {
        'text': 'The Sentiment Proportion for Common Crytocurrency Terms',
        'y':0.95,
        'x':0.48,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='crypto terms', x=0.2, y=0.82, font_size=17, showarrow=False),
                 dict(text='pow', x=0.81, y=0.82, font_size=17, showarrow=False),
                 dict(text='fork', x=0.2, y=0.19, font_size=17, showarrow=False),
                 dict(text='cryptocurrency', x=0.81, y=0.19, font_size=17, showarrow=False)])

fig.write_html('../../images/figure_pies_1.html')

# COMMAND ----------

fig.show()

# COMMAND ----------

# choose four subjects
fig = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])

labels = ['negative', 'neutral', 'positive']

fig.add_trace(go.Pie(labels=labels, values=BTC['count'], marker_colors=['#3D65A5', '#3DA57D', '#F05039'], name="BTC"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=ETH['count'], marker_colors=['#3D65A5', '#3DA57D', '#F05039'], name="ETH"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=polygon['count'], marker_colors=['#3D65A5', '#3DA57D', '#F05039'], name="Polygon"),
              2, 1)
fig.add_trace(go.Pie(labels=labels, values=algorand['count'], marker_colors=['#3D65A5', '#3DA57D', '#F05039'], name="Algorand"),
              2, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    title = {
        'text': 'The Sentiment Proportion for Common Crytocurrency Terms',
        'y':0.95,
        'x':0.48,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='bitcoin', x=0.2, y=0.82, font_size=17, showarrow=False),
                 dict(text='ethereum', x=0.81, y=0.82, font_size=17, showarrow=False),
                 dict(text='polygon', x=0.2, y=0.19, font_size=17, showarrow=False),
                 dict(text='algorand', x=0.81, y=0.19, font_size=17, showarrow=False)])

fig.write_html('../../images/figure_pies_2.html')

# COMMAND ----------

fig.show()
