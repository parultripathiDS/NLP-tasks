
import copy
import numpy as np
import requests
import itertools
import pytextrank
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import json

# initializing the input and output files
topic_results='./results/result.json'
phrase_filepath="./data/record_data_tweet.csv"
all_phrasefile_path="./data/textrank_data_tweet.csv"

# extracting the important phrases from textrank algorithm, currently processing for 3 celebrities (can be more dynamic) <cleaned_df_tweet>
# extracting phrases from all corpus for hashtag task <new_df_tweet>
def extract_phrasesfrom_textrank(corpus):    
    record_data= pd.DataFrame({'sentences': corpus})
    record_data= pd.DataFrame({'id': record_data.index.tolist(),'text': record_data['sentences'].tolist()})
    tweet_items=[]
    for jdict in record_data.to_dict(orient='records'):
        tweet_items.append(jdict)        
        
    new_df_tweet = pd.DataFrame(columns=['text','keywords'])
    path_stage1 = "celebrity1_tweet.json"
    path_stage2 = "celebrity2_tweet.json"
    path_stage3 = "celebrity3_tweet.json"
    for item in tweet_items:
        items_new=[item]
        with open(path_stage1, 'w') as f:
            for graf in pytextrank.parse_doc(items_new):
                f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
               
        graph, ranks = pytextrank.text_rank(path_stage1)
        pytextrank.render_ranks(graph, ranks)
    
        with open(path_stage2, 'w') as f:
            for rl in pytextrank.normalize_key_phrases(path_stage1, ranks):
                f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))
            
        kernel = pytextrank.rank_kernel(path_stage2)
    
        with open(path_stage3, 'w') as f:
            for s in pytextrank.top_sentences(kernel, path_stage1):
                f.write(pytextrank.pretty_print(s._asdict()))
                f.write("\n")
        phrases = ", ".join(set([p for p in pytextrank.limit_keyphrases(path_stage2, phrase_limit=5)]))
        sent_iter = sorted(pytextrank.limit_sentences(path_stage3, word_limit=150), key=lambda x: x[1])
        s = []
    
        for sent_text, idx in sent_iter:
            s.append(pytextrank.make_sentence(sent_text))
    
        graf_text = " ".join(s)
        new_df_tweet = new_df_tweet.append({'text':item.get('text'),'keywords':phrases}, ignore_index=True)  


    celeb_list=['Bradley Cooper','Chris Kyle','Clint Eastwood','bradley cooper','bradley','cooper','chris kyle', 'chris', 'kyle','clint eastwood','clint','eastwood']    
    
    cleaned_df_tweet = pd.DataFrame(columns=['sentences','keywords'])
    for index, row in new_df_tweet.iterrows():
            if any(celeb in row['keywords'] for celeb in celeb_list): 
                cleaned_df_tweet = cleaned_df_tweet.append({'sentences':row['text'],'keywords':row['keywords']}, ignore_index=True)

    cleaned_df_tweet.to_csv(phrase_filepath, sep=',', encoding='utf-8', index=False)
    new_df_tweet.to_csv(all_phrasefile_path, sep=',', encoding='utf-8', index=False)
    return new_df_tweet, cleaned_df_tweet


def topN_colsindexed(df, N):
    a = df.values
    idxtopNpart = np.argpartition(a, -N, axis=1)[:, -1:-N - 1:-1]
    sidx = np.take_along_axis(a, idxtopNpart, axis=1).argsort(1)
    idxtopN = np.take_along_axis(idxtopNpart, sidx[:, ::-1], axis=1)
    c = df.columns.values
    return pd.DataFrame(c[idxtopN], columns=[['Top' + str(i + 1) for i in range(N)]])


# find cosine similarity between feature vector of celebrity and extracted phrases                  
def find_cosine_similarity(cleaned_df_tweet,celeb_list,top):
    feature_array,vectorizer_feature_names,c_vec=doc_term_matrix(cleaned_df_tweet)
    featurenames=celeb_list
    # print(feature_array.shape)
    out_df = pd.DataFrame(data = feature_array, columns = vectorizer_feature_names)
    top_keyword_df=topN_colsindexed(out_df,top)
    keyphrase_list=top_keyword_df.apply(lambda x: x.tolist(), axis=1).tolist()
    all_phrase_list=itertools.chain.from_iterable(keyphrase_list)
    # print (list(all_list))
    all_phrase_list=list(all_phrase_list)
    dict_keywords=dict()
    for index, row in top_keyword_df.iterrows():
        dict_keywords[featurenames[index]]=row.tolist()
    with open(topic_results, 'w') as fp:
        json.dump(dict_keywords, fp)

    return dict_keywords, all_phrase_list

# creating the document term matrix for cleaned tweets with extracted phrases
# removing stopwords and keyword less than 4 length
def doc_term_matrix(cleaned_df_tweet):
    keywords_list = cleaned_df_tweet['keywords'].tolist()
    tweet_list = cleaned_df_tweet['sentences'].tolist()

    cleaned_keywords = []
    for sent_str in keywords_list:
        modified_string = ' '.join([word for word in sent_str.split() if word not in (stopwords.words('english'))])
        modified_string = ' '.join([w for w in modified_string.split() if len(w) > 4])
        cleaned_keywords.append(modified_string)
    keywords_list_clean = []
    for keyword in cleaned_keywords:
        keywords_separated = keyword.split(",")
        #     print(keywords_separated)
        for ke in keywords_separated:
            if ke not in keywords_list_clean:
                # print(ke.replace("'s"," ").replace("'"," "))
                keywords_list_clean.append(ke.strip())
            else:
                pass
    keywords_list_clean = list(filter(None, keywords_list_clean))
    keyword_set = list(set(keywords_list_clean))
    # print(keyword_set)
    c_vec = CountVectorizer(ngram_range=(1, 10), vocabulary=keyword_set)
    X = c_vec.fit_transform(tweet_list)  # needs to happen after fit_transform()
    vocab = c_vec.vocabulary_
    X = X.toarray()
    vectorizer_feature_names = c_vec.get_feature_names()
    out_df = pd.DataFrame(data=X, columns=vectorizer_feature_names)
    featurenames = ['bradly cooper','clint eastwood','chris kyle']
    Cosine_Similarity_Matrix = cosine_similarity(np.transpose(X))
    # vectorizer_feature_names.index('bradly cooper')
    feature_array = ""
    for features in featurenames:
        if len(feature_array) == 0:
            feature_array = np.array([Cosine_Similarity_Matrix[vectorizer_feature_names.index(features)]])
        else:
            feature_array = np.append(feature_array,
                                      [Cosine_Similarity_Matrix[vectorizer_feature_names.index(features)]], axis=0)
    return feature_array,vectorizer_feature_names,c_vec
    # print(feature_array.shape)


# predict the celebrity name on the basis of new tweet (processing 3 celebrity but code can be more generic for more celebrities)
def tweet_prediction(new_tweet, cleaned_df_tweet):
    feature_array, vectorizer_feature_names,c_vec = doc_term_matrix(cleaned_df_tweet)
    new_input_vector = c_vec.transform([new_tweet])
    featurenames = ['bradly cooper', 'clint eastwood', 'chris kyle']
    Cosine_Similarity_Matrix_new_input = cosine_similarity(feature_array, new_input_vector)
    return featurenames[np.argmax(Cosine_Similarity_Matrix_new_input)]

# fetch the synonym from concept net api
def fetch_data_from_concept_net(data, relations, use_proxy = True):
    if isinstance(data, str):
        data = data.split()
    
    # construct URLs for all words in the requested data
    concept_net_request_iris = dict(map(lambda x : (x, "http://api.conceptnet.io/c/en/" + x + '?offset=0&limit=1000'), data))
    concept_net_responses = {}

    # fetch and collate response for each word requested in the data
    for word, concept_net_request_iri in concept_net_request_iris.items():
        print("requesting data for", concept_net_request_iri)
        if use_proxy:
            concept_net_response = requests.get(concept_net_request_iri, proxies = {'http' : "http://10.135.0.29:8080"}).json()
        else:
            concept_net_response = requests.get(concept_net_request_iri).json()
            
        if concept_net_response is not None:
            tmp_response = copy.deepcopy(concept_net_response)
            while 'view' in list(tmp_response.keys()): 
                if 'nextPage' in list(tmp_response['view'].keys()):
                    if use_proxy:
                        tmp_response = requests.get("http://api.conceptnet.io" + tmp_response['view']['nextPage'], proxies = {'http' : "http://10.135.0.29:8080"}).json()
                    else:
                        tmp_response = requests.get("http://api.conceptnet.io" + tmp_response['view']['nextPage']).json()
                    concept_net_response['edges'].extend(tmp_response['edges'])
                else:
                    break
            tmp_response = None
            concept_net_responses[word] = concept_net_response
    
    # filter and aggregate concept net data
    concept_net_data = {}
    for word, concept_net_response in concept_net_responses.items():
        for relation in relations:
            end_tags = [tmp['start']['label'] for tmp in concept_net_response['edges'] if tmp['@type'] == 'Edge' and 
                                                    tmp['rel']['label'] == relation and tmp['start']['language'] == 'en' and tmp['end']['language'] == 'en']
            start_tags = [tmp['end']['label'] for tmp in concept_net_response['edges'] if tmp['@type'] == 'Edge' and 
                                                    tmp['rel']['label'] == relation and tmp['start']['language'] == 'en' and tmp['end']['language'] == 'en']
            
            if len(start_tags) > 0 or len(end_tags) > 0:
                if word not in concept_net_data:
                    concept_net_data[word] = {}
                    
                if relation not in concept_net_data[word]:
                    concept_net_data[word][relation] = []
                    
                concept_net_data[word][relation].extend(list(set(start_tags)))
                concept_net_data[word][relation].extend(list(set(end_tags)))
            
    return concept_net_data