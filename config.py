########################################### START OF PROGRAM ###########################################################
'''
@author: <sauveer.goel@soprasteria.com>
'''
########################################################################################################################
# Implementation: Relationship Microservice Configration File [Ontofy.kit]
# Description:    This file enlists the various configurable parameters for the Relationship Microservice [Ontofy.kit]
########################################################################################################################


###########################################Start of Import Statements###################################################
import logging # to import log levels
###########################################End of Import Statements#####################################################


###########################################Start of Config Class########################################################
class Config(object):
	# specifying filename to store logs
	LOG_FILENAME =                     "logs/TensorflowMicroservice.log" 
	# defining custom log format
	LOG_FORMAT =                       "%(asctime)s:%(msecs)03d %(levelname)s %(filename)s:%(lineno)d %(message)s"
	# defining the date format for logger
	LOG_DATE_FORMAT =                  "%Y-%m-%d %H:%M:%S"
	# defining log levels
	LOG_LEVEL =                        logging.DEBUG
	# defining MongoDB server URL
	MONGODB_SERVER_URL =               "mongo-service.ontofy-kit.svc:27017"
	MONGODB_PROGRESS_COLLECTION = 'relProgress'
#############################################End of Config Class########################################################
