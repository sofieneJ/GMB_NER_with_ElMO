from Elmo_main_server_app import *


from flask import Flask
from flask import request

app = Flask(__name__)
extractor = NERExtractor()
app.logger.info('Main class instanciated')
# graph = tf.get_default_graph()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        app.logger.info(f"Call to classify with POST {request.form['api_paste_data']}")

        ret_text = extractor.predict(input=request.form['api_paste_data']) #, tf_graph=graph
        # app.logger.info('returning...{}'.format(ret_text) )
        return ret_text
    elif request.method == 'GET':
        return 'You sent a GET message'

    print(request)