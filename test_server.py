# from main import *
import requests
import os
import pandas as pd
import json
from sklearn.metrics import classification_report
import numpy as np

# print (os.path.dirname(os.path.abspath(__file__)))


# print (r.text)

def call_extractor_server (input_text, server_session, API_ENDPOINT):
    # data to be sent to api 
    data = { 
            # 'api_option':'paste', 
            'ner_request_text':input_text} 
    
    r = server_session.post(url = API_ENDPOINT, data = data)
#     print (r.content)
    return r.content

def call_retrain_server (input_text, validated_extraction, server_session, API_ENDPOINT):

    # data to be sent to api 
    data = { 
            'validated_extraction':validated_extraction, 
            'retrain_text':input_text
            } 
    
    r = server_session.post(url = API_ENDPOINT, data = data)
#     print (r.content)
    return r.content

def test_GMB_server():

    API_ENDPOINT = "http://52.157.207.101:5000" #Azure
    # API_ENDPOINT = "http://127.0.0.1:5000" #LOCAL
    # API_ENDPOINT = "http://192.168.1.29:5000" #HOME
    
    # your API key here 
    API_KEY = "XXXXXXXXXXXXXXXXX"

    EXTRACT_API_ENDPOINT = API_ENDPOINT +"/extract"
    RETRAIN_API_ENDPOINT = API_ENDPOINT +"/retrain"

    session = requests.Session()
        # your source code here 
    text="""Hundreds of thousands of demonstrators poured into the Hong Kong streets on Sunday in a mass show of support -- marking sixth months of pro-democracy protests and highlighting the resilience of a people who continue to fight for their freedom and autonomy against the Chinese government.
Chanting "Fight for freedom" and "Stand with Hong Kong," the protesters formed a mile-long human snake that winded through blocks from the Causeway Bay shopping district to the Central business zone.
The crowds were reportedly so large that the group was forced to pause at times.
Organizers said 800,000 people participated, although local police didn't have an exact figure.

One of the protesters, however, was nearly hidden from view.

A young woman was seen crawling on her hands and knees on rough streets -- a metaphor for the arduous path and continuous fighting that pro-democracy protesters have faced in order to ensure their eventual freedom.

"This is just the beginning.
We have a long way to run," Louisa Yiu, an engineer and protester, told the Guardian.
The crawling protester also dragged bricks and empty soda cans on a string behind her -- another metaphor for the weight they've been carrying -- which excited fellow protesters who were heard yelling "Go for it!"

"Her performance art is about the difficulty, or the repetitiveness, of demonstrations," said one of her friends, who walked alongside and identified herself by her surname, Chan. "This is really a long-term struggle."

Marchers were captured holding up five fingers, a symbol for the protest movement's five demands.
They include democratic elections and an investigation into the actions of police throughout the last six months of protests, according to the Guardian.
The belief was that the protester movement would cease over time as it enters the seventh month.
It hasn't."""
    

    str_ret = call_extractor_server(text, session, EXTRACT_API_ENDPOINT)
    print (f"extraction results \n {json.loads(str_ret)}" )

    validation_results = """{"DocumentId":"hongkong_v.pdf","ResultsVersion":1,"ResultsDocument":{"Bounds":{"StartPage":0,"PageCount":1,"TextStartIndex":0,"TextLength":1852},"Language":"","DocumentGroup":"NewsArticle","DocumentCategory":"Politics","DocumentTypeId":"NewsArticle.Politics.article","DocumentTypeName":"article","DocumentTypeDataVersion":0,"DataVersion":1,"DocumentTypeSource":"Automatic","DocumentTypeField":{"Components":[],"Value":"article","Reference":{"TextStartIndex":0,"TextLength":0,"Tokens":[]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},"Fields":[{"FieldId":"NewsArticle.Politics.article.Person","FieldName":"Person","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Louisa Yiu,","Reference":{"TextStartIndex":1001,"TextLength":11,"Tokens":[{"TextStartIndex":1001,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[397.03,333.7382,32.4753,11.04],[397.03,368.819,17.2113,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Chan.","Reference":{"TextStartIndex":1448,"TextLength":5,"Tokens":[{"TextStartIndex":1448,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[544.06,419.3624,27.7988,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.TimeOrDate","FieldName":"TimeOrDate","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Sunday","Reference":{"TextStartIndex":76,"TextLength":6,"Tokens":[{"TextStartIndex":76,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[81.98,457.3367,35.8138,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.Organization","FieldName":"Organization","FieldType":"Text","IsMissing":false,"DataSource":"Manual","Values":[{"Components":[],"Value":"Guardian.","Reference":{"TextStartIndex":1049,"TextLength":9,"Tokens":[{"TextStartIndex":1049,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[418.03,90.024,46.3146,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Guardian.","Reference":{"TextStartIndex":1735,"TextLength":9,"Tokens":[{"TextStartIndex":1735,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[649.08,359.5166,46.2798,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeographicEntity","FieldName":"GeographicEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Hong Kong","Reference":{"TextStartIndex":55,"TextLength":9,"Tokens":[{"TextStartIndex":55,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[81.98,349.99,26.0323,11.04],[81.98,378.67,25.3147,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Hong Kong,'","Reference":{"TextStartIndex":333,"TextLength":11,"Tokens":[{"TextStartIndex":333,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[165.98,288.3749,26.0433,11.04],[165.98,317.0458,30.6139,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Causeway Bay","Reference":{"TextStartIndex":427,"TextLength":12,"Tokens":[{"TextStartIndex":427,"TextLength":12,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[186.98,316.6704,49.7462,11.04],[186.98,369.0773,17.8517,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeopoliticalEntity","FieldName":"GeopoliticalEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Chinese","Reference":{"TextStartIndex":268,"TextLength":7,"Tokens":[{"TextStartIndex":268,"TextLength":7,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[144.98,90.024,38.5517,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1}]}}"""
    str_ret = call_retrain_server(text, validation_results, session, RETRAIN_API_ENDPOINT)
    try:
        print (f"retraining results \n {json.loads(str_ret)}")
    except:
        print (f"retraining return \n {str_ret}")

    str_ret = call_extractor_server(text, session, EXTRACT_API_ENDPOINT)
    print (f"extraction results \n {json.loads(str_ret)}" )

def test_FB_server():

    API_ENDPOINT = "http://52.157.207.101:5000" #Azure
    # API_ENDPOINT = "http://127.0.0.1:5000" #LOCAL
    # API_ENDPOINT = "http://192.168.1.29:5000" #HOME
    
    # your API key here 
    API_KEY = "XXXXXXXXXXXXXXXXX"

    EXTRACT_API_ENDPOINT = API_ENDPOINT +"/extract"
    RETRAIN_API_ENDPOINT = API_ENDPOINT +"/retrain"

    session = requests.Session()
        # your source code here 
    text="""I would like to travel from Paris to Berlin from 1st January 2019 to 24th January 2020.
I would like to fly from Rome to Munich from 2nd March 2020 to 24th June 2020.
I want to book a flight to London from Paris from 3rd February 2020 to 15th Mai 2020.
I want to travel to Bucharest from Madrid from 06/01/2020 to 02/04/2020.
I am willing to fly from Bucharest to Barcelona from 03/03/2020 to 07/07/2020.
I wish to travel from Porto to Marseille from 4th April 2020 to 6th August 2020.
"""
    

    str_ret = call_extractor_server(text, session, EXTRACT_API_ENDPOINT)
    print (f"extraction results \n {json.loads(str_ret)}" )

    # validation_results = """{"DocumentId":"hongkong_v.pdf","ResultsVersion":1,"ResultsDocument":{"Bounds":{"StartPage":0,"PageCount":1,"TextStartIndex":0,"TextLength":1852},"Language":"","DocumentGroup":"NewsArticle","DocumentCategory":"Politics","DocumentTypeId":"NewsArticle.Politics.article","DocumentTypeName":"article","DocumentTypeDataVersion":0,"DataVersion":1,"DocumentTypeSource":"Automatic","DocumentTypeField":{"Components":[],"Value":"article","Reference":{"TextStartIndex":0,"TextLength":0,"Tokens":[]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},"Fields":[{"FieldId":"NewsArticle.Politics.article.Person","FieldName":"Person","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Louisa Yiu,","Reference":{"TextStartIndex":1001,"TextLength":11,"Tokens":[{"TextStartIndex":1001,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[397.03,333.7382,32.4753,11.04],[397.03,368.819,17.2113,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Chan.","Reference":{"TextStartIndex":1448,"TextLength":5,"Tokens":[{"TextStartIndex":1448,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[544.06,419.3624,27.7988,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.TimeOrDate","FieldName":"TimeOrDate","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Sunday","Reference":{"TextStartIndex":76,"TextLength":6,"Tokens":[{"TextStartIndex":76,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[81.98,457.3367,35.8138,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.Organization","FieldName":"Organization","FieldType":"Text","IsMissing":false,"DataSource":"Manual","Values":[{"Components":[],"Value":"Guardian.","Reference":{"TextStartIndex":1049,"TextLength":9,"Tokens":[{"TextStartIndex":1049,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[418.03,90.024,46.3146,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Guardian.","Reference":{"TextStartIndex":1735,"TextLength":9,"Tokens":[{"TextStartIndex":1735,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[649.08,359.5166,46.2798,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeographicEntity","FieldName":"GeographicEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Hong Kong","Reference":{"TextStartIndex":55,"TextLength":9,"Tokens":[{"TextStartIndex":55,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[81.98,349.99,26.0323,11.04],[81.98,378.67,25.3147,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Hong Kong,'","Reference":{"TextStartIndex":333,"TextLength":11,"Tokens":[{"TextStartIndex":333,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[165.98,288.3749,26.0433,11.04],[165.98,317.0458,30.6139,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Causeway Bay","Reference":{"TextStartIndex":427,"TextLength":12,"Tokens":[{"TextStartIndex":427,"TextLength":12,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[186.98,316.6704,49.7462,11.04],[186.98,369.0773,17.8517,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeopoliticalEntity","FieldName":"GeopoliticalEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Chinese","Reference":{"TextStartIndex":268,"TextLength":7,"Tokens":[{"TextStartIndex":268,"TextLength":7,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[144.98,90.024,38.5517,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1}]}}"""
    # str_ret = call_retrain_server(text, validation_results, session, RETRAIN_API_ENDPOINT)
    # try:
    #     print (f"retraining results \n {json.loads(str_ret)}")
    # except:
    #     print (f"retraining return \n {str_ret}")

    # str_ret = call_extractor_server(text, session, EXTRACT_API_ENDPOINT)
    # print (f"extraction results \n {json.loads(str_ret)}" )

    

if __name__=='__main__':
    # test_GMB_server()
    test_FB_server()
    
    # my_list = [1,2,3,4]
    # np_list= np.array(my_list)
    # my_array = np.empty((0,4))
    # my_array = np.vstack((my_array, np_list))
    # my_array = np.vstack((my_array, np_list))

    # print (np.vstack((np_list, np.array([]))))
    # print (my_array)
    # word1 = "hello"
    # word2 = "there"
    # print (' '.join([word1, word2]))
    # print (len(' '.join([word1])))
    # word="aaa\"bbb"
    # print(word.replace("\"","\\\""))

    # my_list = [np.float32(0.01), np.float32(0.91)]
    # my_dic = {
    #     "liste":my_list
    # }
    # print (isinstance(my_dic["liste"], list))
    # print (isinstance(my_dic["liste"][0], np.float32))
    # print (type(my_dic["liste"][0]))


    # from helpers import *
    # print (json.dumps(my_dic, cls=NumpyEncoder, sort_keys=True, indent=2))
    
    
