from math import ceil

import codecs
import os
from lxml import etree
import pandas as pd




def check_positive_int(value):
    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    
    return ivalue



def check_positive_float(value):
    fvalue = float(value)

    if fvalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    
    return fvalue



def check_dir_exists(dir_path):
    dir_path_str = str(dir_path)

    if not os.path.isdir(dir_path_str):
        raise argparse.ArgumentTypeError("\'%s\' is not a valid directory" % dir_path_str)

    return dir_path_str



def check_is_file(file_path):
    file_path_str = str(file_path)

    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError("\'%s\' is not a valid file" % file_path_str)

    return file_path



def read_amazon_data(json_path, lines_to_read=None):
    json_file = open(json_path, 'r')

    i = 0
    df = {}

    for d in json_file.readlines():
        curr_dict = eval(d)
        df[i] = curr_dict
        i += 1

        if (lines_to_read != None) and i == lines_to_read:
            break
    
    json_file.close()

    return pd.DataFrame.from_dict(df, orient='index')



def read_review_data(dir_path, max_lines_to_read = 2000):
    # Did so in order to just get the same
    # reasonable number of reviews in each category
    # otherwise the CPU gets Killed
    
    columns   = ['Title', 'Review', 'Rating']
    instances = []

    product_categories = sorted(os.listdir(dir_path))

    for product_category in product_categories:
        # I exclude books category for now because the file is too big.
        if product_category in ['books', 'stopwords', 'summary.txt']:
            continue

        print(product_category)
        file_path = os.path.join(dir_path, product_category, 'all.review')
        file_obj  = codecs.open(file_path, encoding='utf-8', mode='r', errors='ignore')

        data = file_obj.read()
        file_obj.close()

        # Adding an adhoc root element in order to be able to
        # parse the files
        data = '<myroot>' + data + '</myroot>'

        parser = etree.XMLParser(recover=True)
        root   = etree.fromstring(data, parser=parser)

        reviews_read = 0
        for review in root:
            assert(review.tag == 'review')

            if review.find('rating') == None:
                continue
            
            if review.find('title') == None:
                continue

            if review.find('review_text') == None:
                continue
            
            curr_title  = review.find('title').text.strip()
            curr_review = review.find('review_text').text.strip()
            curr_rating = review.find('rating').text.strip()

            instances.append([curr_title, curr_review, curr_rating])

            reviews_read += 1
            if reviews_read == max_lines_to_read:
                break
    
    return pd.DataFrame(instances, columns=columns)



def read_imdb_data(dir_path):
    columns  = ['Review', 'Rating']
    datasets = []
    valences = ['neg', 'pos']

    for dataset in ['train', 'test']:
        dataset_dir = os.path.join(dir_path, dataset)
        curr_dataset = []

        for valence in valences:
            valence_dir = os.path.join(dataset_dir, valence)

            instance_files = os.listdir(valence_dir)

            for instance_filename in instance_files:
                # Get the rating of the file through its filename
                curr_rating = int(instance_filename.split('_')[1].split('.')[0])

                instance_file_path = os.path.join(valence_dir, instance_filename)
                instance_file      = open(instance_file_path, 'r')

                curr_review = instance_file.read()
                instance_file.close()

                curr_dataset.append([curr_review, curr_rating])

            print('Read /' + dataset + '/' + valence)

        datasets.append(curr_dataset)

    train_instances, test_instances = datasets

    return pd.DataFrame(train_instances, columns=columns), \
            pd.DataFrame(test_instances, columns=columns)



def __max_5_rating_to_sentiment(rating):
    if rating < 3.0:
        return 0
    elif rating > 3.0:
        return 1

    return 2

def __max_10_rating_to_sentiment(rating):
    if rating <= 4.0:
        return 0
    elif rating >= 7.0:
        return 1

    return 2

"""
It will convert the ratings to a "numerical" sentiment with the following ways
depending the scale:

Overall 1-2 -> 0(=Negative Sentiment)
Overall 3   -> 2(=Neutral Sentiment)
Overall 4-5 -> 1(=Positive Sentiment)

OR

Overall <= 4     -> 0(=Negative Sentiment)
4 < Overall < 7  -> 2(=Neutral Sentiment)
Overall >= 7     -> 1(=Positive Sentiment)

"""
def convert_ratings_to_sentiments(overall_ratings):
    max_rating = max(list(overall_ratings.apply(lambda x: int(x))))

    assert(max_rating in [5,10])

    if max_rating == 5:
        return overall_ratings.apply(lambda x: __max_5_rating_to_sentiment(x))
    else:
        return overall_ratings.apply(lambda x: __max_10_rating_to_sentiment(x))
