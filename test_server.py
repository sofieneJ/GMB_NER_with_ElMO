# from main import *
import requests
import os
import pandas as pd
import json
from sklearn.metrics import classification_report
import numpy as np

# print (os.path.dirname(os.path.abspath(__file__)))


# print (r.text)

def call_extractor_server (input_text, server_session):
    # defining the api-endpoint  
    
    # API_ENDPOINT = "http://52.157.207.101:5000/extract" #Azure
    API_ENDPOINT = "http://127.0.0.1:5000/extract" #LOCAL
    # API_ENDPOINT = "http://192.168.1.29:5000/extract" #HOME
    
    # your API key here 
    API_KEY = "XXXXXXXXXXXXXXXXX"
    
    
    # data to be sent to api 
    data = { 
            # 'api_option':'paste', 
            'api_paste_data':input_text} 
    
    # sending post request and saving response as response object 
    
    r = server_session.post(url = API_ENDPOINT, data = data)
#     print (r.content)
    return r.content

def test_server():

    session = requests.Session()
        # your source code here 
    sample = """                Some Title
An Australian academic freed by the Taliban in a prisoner swap has spoken of his "long and tortuous ordeal" as a hostage in Afghanistan.

Malta police arrested one of the country's most prominent businessmen on Wednesday in connection with an investigation into the murder of journalist Daphne Caruana Galizia."""
    str_ret = call_extractor_server(sample,session)
    print (json.loads(str_ret))
    

if __name__=='__main__':
    # test_server()
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
    word="aaa\"bbb"
    print(word.replace("\"","\\\""))
    
    
