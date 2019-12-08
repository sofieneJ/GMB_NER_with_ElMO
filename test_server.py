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

def test_server():

    # API_ENDPOINT = "http://52.157.207.101:5000" #Azure
    API_ENDPOINT = "http://127.0.0.1:5000" #LOCAL
    # API_ENDPOINT = "http://192.168.1.29:5000" #HOME
    
    # your API key here 
    API_KEY = "XXXXXXXXXXXXXXXXX"

    EXTRACT_API_ENDPOINT = API_ENDPOINT +"/extract"
    RETRAIN_API_ENDPOINT = API_ENDPOINT +"/retrain"

    session = requests.Session()
        # your source code here 
    text="""Indian police are being declared heroes Friday after they shot and killed four men suspected of raping and killing a young woman in southern India just a week earlier.

Last Thursday, the remains of a 27-year-old veterinarian were discovered by a passerby at an underpass in the town of Shadnager, near Hyderabad, after she left her scooter at a toll booth for a medical visit the night before.

The four men -- between the ages of 20 and 26 -- allegedly deflated her scooter and took her to a truck yard to repair it. Soon after, they assaulted and suffocated the woman before burning her body, according to Sky News."""
    

    str_ret = call_extractor_server(text, session, EXTRACT_API_ENDPOINT)
    print (f"extraction results \n {json.loads(str_ret)}" )

    validation_results = """{"DocumentId":"india_news_v.pdf","ResultsVersion":1,"ResultsDocument":{"Bounds":{"StartPage":0,"PageCount":1,"TextStartIndex":0,"TextLength":618},"Language":"","DocumentGroup":"NewsArticle","DocumentCategory":"Politics","DocumentTypeId":"NewsArticle.Politics.article","DocumentTypeName":"article","DocumentTypeDataVersion":0,"DataVersion":1,"DocumentTypeSource":"Automatic","DocumentTypeField":{"Components":[],"Value":"article","Reference":{"TextStartIndex":0,"TextLength":0,"Tokens":[]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},"Fields":[{"FieldId":"NewsArticle.Politics.article.Person","FieldName":"Person","FieldType":"Text","IsMissing":true,"DataSource":"Automatic","Values":[],"DataVersion":0},{"FieldId":"NewsArticle.Politics.article.TimeOrDate","FieldName":"TimeOrDate","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Friday","Reference":{"TextStartIndex":40,"TextLength":6,"Tokens":[{"TextStartIndex":40,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[74.18,250.5975,27.313,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Thursday,","Reference":{"TextStartIndex":174,"TextLength":9,"Tokens":[{"TextStartIndex":174,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[133.7,92.4149,43.895,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"night","Reference":{"TextStartIndex":381,"TextLength":5,"Tokens":[{"TextStartIndex":381,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[162.62,89.5114,22.93,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.Organization","FieldName":"Organization","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Sky News.","Reference":{"TextStartIndex":609,"TextLength":9,"Tokens":[{"TextStartIndex":609,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[236.57,130.0613,14.9812,11.04],[236.57,147.5707,27.5007,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeographicEntity","FieldName":"GeographicEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"southern India","Reference":{"TextStartIndex":132,"TextLength":14,"Tokens":[{"TextStartIndex":132,"TextLength":14,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[88.7,204.3274,40.6161,11.04],[88.7,247.3944,22.1352,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Shadnager,","Reference":{"TextStartIndex":287,"TextLength":10,"Tokens":[{"TextStartIndex":287,"TextLength":10,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[148.1,137.6126,50.205,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Hyderabad,","Reference":{"TextStartIndex":303,"TextLength":10,"Tokens":[{"TextStartIndex":303,"TextLength":10,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[148.1,212.9998,51.866,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeopoliticalEntity","FieldName":"GeopoliticalEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Indian","Reference":{"TextStartIndex":0,"TextLength":6,"Tokens":[{"TextStartIndex":0,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[74.18,72.024,27.9091,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1}]}}"""
    str_ret = call_retrain_server(text, validation_results, session, RETRAIN_API_ENDPOINT)
    try:
        print (f"retraining results \n {json.loads(str_ret)}")
    except:
        print (f"retraining return \n {str_ret}")

    str_ret = call_extractor_server(text, session, EXTRACT_API_ENDPOINT)
    print (f"extraction results \n {json.loads(str_ret)}" )

    

if __name__=='__main__':
    test_server()
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
    
    
