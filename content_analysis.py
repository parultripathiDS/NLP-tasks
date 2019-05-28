
from pathlib import Path
import docx2txt
import pandas as pd
from itertools import chain, combinations
import numpy as np
from copy import deepcopy
from term_extraction import extract_phrases,stitch_related_phrases,fetch_data_from_concept_net,extract_phrasesfrom_textrank
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re


# In[]
def fetch_vocabulary(vocab_terms):
    relations_to_extract = ["Synonym"]
    vocabulary = fetch_data_from_concept_net(vocab_terms, relations_to_extract, use_proxy =True)    
    return vocabulary

# In[]

def extract_terms(new_data): 
    important_phrases_from_adjacency_list = []
    #noun_terms = {}
    #important_terms_from_idf = {}
    
        
    if new_data is not None:
        #important_terms_from_idf[file.name] = extract_important_features_using_tf_idf(tmp_data, ngrams= 1, term_count = 5)
        #important_phrases_from_adjacency_list = extract_phrases(tmp_data, requested_terms= 500, term_freq_threshold= 2, spell_correction=False)        
        important_phrases_from_adjacency_list = extract_phrasesfrom_textrank(new_data)  
                
        #noun_terms[file.name] = list(set((filter_nouns(list(set(chain(*[list(chain(*important_terms_from_idf[file.name])), important_phrases_from_adjacency_list[file.name]])))).values())))
        
        #print("\n\n\n\n doc : {0}, \n\nnouns : {1}, \n\nimportant_terms from adjacency: {2}, \
        #      \n\ntf-idf : {3} \n\n".format(file.name, noun_terms[file.name], important_phrases_from_adjacency_list[file.name],  
        #      important_terms_from_idf[file.name]))
        
        print(important_phrases_from_adjacency_list)
                
    #return important_phrases_from_adjacency_list, noun_terms, important_terms_from_idf
    return important_phrases_from_adjacency_list


def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def text_preprocessing(textdata):    
    sentences = re.split(r'\n', textdata)
    new_sents=[]   
    for sentences in sentences:    
        text = re.sub(r'http?:\/\/.*[\r\n]*', '', sentences, flags=re.MULTILINE)
        text = re.sub('[!@#$:).;,?&]', '', text)
        text=  re.sub('  ', ' ', text)
        text=deEmojify(text)
        new_sents.append(text)
    
    new_sents=list(set(new_sents))
    new_sents = [x.lower() for x in new_sents]    
    new_sents = filter(None, new_sents) # fastest
    new_sents = filter(bool, new_sents) # fastest
    new_sents = filter(len, new_sents)  # a bit slower
    new_sents = filter(lambda item: item, new_sents) 
    new_sents = list(filter(None, new_sents)) 
    return new_sents


# find Entities and keywords for the respective documents
tmp_data = None
filename="C:/JupyterFiles/assignment_test/tweets.txt"
with open(filename,encoding="utf8") as f:
    text = f.read()

new_data = text_preprocessing(text)
phrases = extract_terms(new_data)

all_tags = {tag.strip("#") for tag in text.split() if tag.startswith("#")}
all_tags = [re.sub('[^a-zA-Z0-9]+', '', _) for _ in all_tags]
all_tags = list(set(all_tags))
all_tags = [x.lower() for x in all_tags]   
synonym = fetch_vocabulary(phrases)

list_of_bigram_docs = {}
for tag in all_tags:    
    # Get a list of matches ordered by score, default limit to 5
    #process.extract(tag, my_list)
    # [('Barack H Obama', 95), ('Barack H. Obama', 95), ('B. Obama', 85)]
    finalvalue = process.extractOne(tag, phrases)    
   
    if finalvalue is not None:        
        if len(finalvalue[0]) == len(tag)+1 and finalvalue[1] > 90:
            list_of_bigram_docs[tag] = finalvalue  
    
list_of_bigram_docs
    # ('Barack H Obama', 95) 
    
    
    
