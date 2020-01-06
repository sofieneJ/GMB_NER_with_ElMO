from Elmo_main_server_app import *


from flask import Flask
from flask import request


app = Flask(__name__)
extractor = NERExtractor(appName="GMB", bLoadFromRetrained=False)  
app.logger.info('Main class instanciated')


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        app.logger.info(f"Call to extract with POST: \n {request.form['ner_request_text']}")

        ret_text = extractor.predict(input=request.form['ner_request_text'])
        # app.logger.info('returning...{}'.format(ret_text) )
        return ret_text
    elif request.method == 'GET':
        return 'You sent a GET message'

    print(request)

@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        app.logger.info(f"Call to retrain with POST: \n {request.form['retrain_text']}")

        ret_text = extractor.retrain(text=request.form['retrain_text'], validation_results=request.form['validated_extraction'])
        # app.logger.info('returning...{}'.format(ret_text) )
        return ret_text
    elif request.method == 'GET':
        return 'You sent a GET message'

    print(request)


