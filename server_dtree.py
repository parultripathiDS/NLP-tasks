from dragnn.wrapper import *					 #
from pymongo import MongoClient                  # to perform operation related to MongoDB
import json                                      # to convert the results into json dictionaries and 
                                                 #     write them in MongoDB
from flask import Flask                          # to deploy as a flask service
from flask import request                        # to be able to read the incoming requests
from flask import jsonify                        # to parse the response as JSON
from flask import abort                          # to close the connection with the client with an error code
from flask_cors import CORS                      # to enable Cross-Origin Resource Sharing
import logging                                   # to perform logging
from config import Config                        # importing the configrable variables
import string

# Initializing Flask Application
app = Flask(__name__)          # 'app' refers to initialized Flask App
CORS(app)                      # to enable CORS for the services
app.config.from_object(Config) # attaching the configurations to the flask app

# Configuring Logs
logging.basicConfig(filename=app.config['LOG_FILENAME'], filemode='a',format=app.config['LOG_FORMAT']\
    , datefmt=app.config['LOG_DATE_FORMAT'], level=app.config['LOG_LEVEL'])


"""
    Service to accept a sentence and output the parsed tree
    -Only caters to a single sentence

    Return a Dependency tree in list of dictionary format
    Output :
	[{
    word, original_word : the word in the sentence
    label : dependency label 
    tails : empty list to store index of tail_labels
    tail_labels : empty list to store labels of all the tail words
    start : start position in the string
    end : ending position in the string
    head : index of head tag
    tag : attributes of the word as per dependency parsing
    break_level : separation with previous word
	}]

"""
@app.route("/annotate_sentence", methods = ['POST'])
def annotate_sentence():
    # get the json body from the request
    input = request.json
    # extract the sentence to be parsed
    sentence = input['sentence']
    sentence = str(sentence.encode('ascii', 'ignore'))
    logging.debug("sentence received for parsing : {0}".format(sentence))

    annotated_sentence = syntaxnet_instance.annotate(sentence)
    #converting the sentence object to list of dictionary
    parsed_tree = [{"word" : annotated_sentence.token[index].word, "tails": [], "tail_lables": [], "original_word": annotated_sentence.token[index].word, "start" : annotated_sentence.token[index].start, "end" : annotated_sentence.token[index].end, "head" : annotated_sentence.token[index].head, "tag" : annotated_sentence.token[index].tag, "category" : annotated_sentence.token[index].category, "label" : annotated_sentence.token[index].label, "break_level" : annotated_sentence.token[index].break_level,} for index in range(0, len(annotated_sentence.token))]
    result = {"tree" : parsed_tree}
    return jsonify(result)

	
	# Initializing  the flask application with port as 4892 and open to any incoming connection(0.0.0.0)
if __name__ == '__main__':
    syntaxnet_instance = SyntaxNet()
    app.run(host="0.0.0.0",port=4892,threaded=True,use_reloader=False)
