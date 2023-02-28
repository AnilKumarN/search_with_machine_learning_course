import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import nltk

nltk.download('stopwords')
import string

import pandas as pd
from nltk.stem.snowball import SnowballStemmer


def transform_name(product_name):
     product_name = product_name.lower()

     translator = product_name.maketrans(string.punctuation, ' '*len(string.punctuation))
     product_name = product_name.translate(translator)
     product_name = ' '.join(product_name.split())

     stemmer = SnowballStemmer('english', ignore_stopwords=True)
     product_name = stemmer.stem(product_name)

     return product_name

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

#Setting category depth
general.add_argument("--category_level_depth", default=3, type=int, help="The number of levels we parse into the category")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
min_products = args.min_products
sample_rate = args.sample_rate
category_level_depth = args.category_level_depth
category_list = []

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                      # Choose last element in categoryPath as the leaf categoryId

                      if len(child.find('categoryPath'))<category_level_depth:
                          cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                      else:
                          cat = child.find('categoryPath')[category_level_depth - 1][0].text

                      name = child.find('name').text.replace('\n', ' ')
                      category_list.append(dict([('category', cat),('product_name', transform_name(name))]))

    category_list_df = pd.DataFrame(category_list)

    if min_products>0:
        category_list_df=category_list_df.groupby("category").filter(lambda x: len(x) > min_products)

    for _, row in category_list_df.iterrows():
            output.write("__label__%s %s\n" % (row['category'], row['product_name']))

