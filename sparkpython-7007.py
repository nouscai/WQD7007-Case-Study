#!/usr/bin/env python
# coding: utf-8

# In[43]:


get_ipython().system('pip install pyspark')

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Disneyland Reviews") \
    .getOrCreate()

file_path = '/Users/nous/Desktop/DisneylandReviews.csv'


df = spark.read.csv(file_path, header=True, inferSchema=True)

df.show(5)


# In[44]:


from pyspark.sql.functions import split

split_column = split(df['Year_Month'], '-')

df = df.withColumn('Year', split_column.getItem(0))
df = df.withColumn('Month', split_column.getItem(1))

df.show()


# In[45]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnull

# 计算每列的空值数量
null_counts = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])

# 显示空值情况
null_counts.show()

total_rows = df.count()
print(total_rows)


# In[46]:


df = df.withColumn("Year_Month", when(df["Year_Month"] == "missing", 0).otherwise(df["Year_Month"]))


df = df.withColumn("Year", when(df["Year"] == "missing", 0).otherwise(df["Year"]))


df = df.withColumn("Month", when(df["Month"].isNull(), 0).otherwise(df["Month"]))


null_counts = df.select([count(when(df[c].isNull(), c)).alias(c) for c in df.columns])


null_counts.show()

total_rows = df.count()

print(total_rows)


# In[47]:


# Drop the review_text column
df_without_review_text = df.drop('review_text')

# Show the DataFrame without the review_text column
df_without_review_text.show(5)

df_without_review_text.write.csv('/Users/nous/Desktop/Disneylandhive.csv', header=True)


# In[48]:


df_without_review_text.coalesce(1).write.csv('/Users/nous/Desktop/Disneylandhive1', header=True)


# In[49]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, concat_ws, regexp_replace
from pyspark.ml.feature import StopWordsRemover

spark = SparkSession.builder \
    .appName("Extract Review Text") \
    .getOrCreate()

punctuation_pattern = "[^\\w\\s]"

review_text_df = df.withColumn("review_text", regexp_replace(col("review_text"), punctuation_pattern, ""))

review_text_df = review_text_df.withColumn("words", split(col("review_text"), " "))

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

review_text_df = remover.transform(review_text_df)

review_text_df = review_text_df.withColumn("filtered_text", concat_ws(" ", "filtered_words"))

collected_reviews = review_text_df.select("filtered_text").rdd.flatMap(lambda x: x).collect()

output_path = '/Users/nous/Desktop/output.txt'
with open(output_path, 'w') as file:
    for review in collected_reviews:
        file.write(review + "\n")

