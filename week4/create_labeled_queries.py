import argparse
import csv
import os
import string
import xml.etree.ElementTree as ET

# Useful if you want to perform stemming.
import nltk
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1000,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

#Convert queries to lowercase, and optionally implement other normalization, like stemming.
def transform_query(query):
    query = query.lower()

    translator = query.maketrans(string.punctuation, ' '*len(string.punctuation))
    query = query.translate(translator)
    query = ' '.join(query.split())

    query = stemmer.stem(query)

    return query

#Roll up categories to ancestors to satisfy the minimum number of queries per category.
def roll_upto_parentcategory(cat):
    if cat == root_category_id:
        return cat
    else:
        return parents_df[parents_df.category == cat]['parent'].values[0]


df = df[df['category'].isin(categories)]
df['query'] = df['query'].apply(transform_query)
df = df.dropna()
df['count_per_category'] = df.groupby('category')['query'].transform(len)
df['parent_category'] = df['category'].apply(roll_upto_parentcategory)

conditions = [
    (df['count_per_category'] <= min_queries),
    (df['count_per_category'] > min_queries)
    ]
values = [df['parent_category'], df['category']]
df['category'] = np.select(conditions, values)
print(df.category.nunique())




# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
