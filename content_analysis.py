from os import path
from term_extraction import fetch_data_from_concept_net, extract_phrasesfrom_textrank, find_cosine_similarity, \
    tweet_prediction
from fuzzywuzzy import process
import re
import pandas as pd
import json
import argparse
from dask.dataframe.multi import required

# initializing the input files
phrase_filepath = "./data/record_data_tweet.csv"
tweet_file = "./data/tweets.txt"
all_phrasefile_path="./data/textrank_data_tweet.csv"
#all_phrasefile_path="./data/record_data_tweet.csv"

# initializing the output file
synonyms_path = './results/synonyms.json'
hash_tag_keyword_path = './results/hashtag_keyword.json'


# to extract the synonym from given phrases
def fetch_vocabulary(vocab_terms):
    relations_to_extract = ["Synonym"]
    vocabulary = fetch_data_from_concept_net(vocab_terms, relations_to_extract, use_proxy=True)
    return vocabulary


# remove Emoji from tweets
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')


# clean the tweet data
def text_preprocessing(textdata,tag):
    sentences = re.split(r'\n', textdata)
    new_sents = []
    for sentences in sentences:
        text = re.sub(r'http?:\/\/.*[\r\n]*', '', sentences, flags=re.MULTILINE)
        if tag:
            text = re.sub('[!@$:).;,?&]', '', text)
        else:
            text = re.sub('[!@#$:).;,?&]', '', text)
        text = re.sub('  ', ' ', text)
        text = deEmojify(text)
        new_sents.append(text)

    new_sents = list(set(new_sents))
    new_sents = [x.lower() for x in new_sents]
    new_sents = filter(None, new_sents)  # fastest
    new_sents = filter(bool, new_sents)  # fastest
    new_sents = filter(len, new_sents)  # a bit slower
    new_sents = filter(lambda item: item, new_sents)
    new_sents = list(filter(None, new_sents))
    return new_sents


# command line arguments from user
# python content_analysis.py -t topics -c "bradley cooper, clint eastwood, chris kyle" -n 20
# python content_analysis.py -t prediction -s "AmericanSniper Chris Kyle's widow This movie will be how my kids remember their dad"
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--task", required=True,
                    help="task name", type=str)
    ap.add_argument("-s", "--sentence",
                    help="sentence to predict", type=str)
    ap.add_argument("-c", "--celeb",
                    help="celeb name list", type=str)
    ap.add_argument("-n", "--n",
                    help="number of top keywords", type=int)

    args = vars(ap.parse_args())

    # read the tweet file
    tmp_data = None
    with open(tweet_file, encoding="utf8") as f:
        text = f.read()

    # check if phrases already exists then load the phrases files else create the phrases using textRank algorithm
    if path.exists(phrase_filepath):
        pass
    else:
        new_data = text_preprocessing(text,False)
        new_df_tweet, cleaned_df_tweet = extract_phrasesfrom_textrank(new_data)
        cleaned_df_tweet.to_csv(phrase_filepath, sep=',', encoding='utf-8', index=False)


    # read the phrases file extracted from textrank algorithm
    cleaned_df_tweet = pd.read_csv(phrase_filepath)

    # case1 - find the celebrity name -- Input: New Tweet  Output: name of celebrity related to given tweet
    # case2 - find the trending topics about celebrity -- Input: list of celebrity name and number of topics  Output: json (celebrity and topics)
    if args["task"] == "prediction":
        new_tweet = args["sentence"]
        print(tweet_prediction(new_tweet, cleaned_df_tweet))

    elif args["task"] == "topics":
        str_celeb = args["celeb"]
        celeb_list = str_celeb.split(", ")
        top = args["n"]
        dict_keywords, all_phrase_list = find_cosine_similarity(cleaned_df_tweet, celeb_list, top)
        synonym = fetch_vocabulary(all_phrase_list)
        with open(synonyms_path, 'w') as fp:
            json.dump(synonym, fp)

    # code to break the twitter hashtags into proper words and saving the file in hashtag_keyword.json
    elif args["task"] == "tag":
        text=text_preprocessing(text,True)
        text=" ".join(text)
        all_tags = {tag.strip("#") for tag in text.split() if tag.startswith("#")}
        all_tags = [re.sub('[^a-zA-Z]+', '', _) for _ in all_tags]
        all_tags = list(set(all_tags))
        all_tags = [x.lower() for x in all_tags]
        if path.exists(all_phrasefile_path):
            all_phrase_df = pd.read_csv(all_phrasefile_path)
            phrases = all_phrase_df['keywords'].tolist()

            list_of_bigram_docs = {}
            for tag in all_tags:
                finalvalue = process.extractOne(tag, phrases)
                if finalvalue is not None:
                    if len(finalvalue[0]) == len(tag) + 1 and finalvalue[1] > 90:
                        list_of_bigram_docs[tag] = finalvalue

            with open(hash_tag_keyword_path, 'w') as fp:
                json.dump(list_of_bigram_docs, fp)

        else:
            print("File"+all_phrasefile_path+" doesn't exist.")



