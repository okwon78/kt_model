import os
import shutil
from multiprocessing import Process
from os import path

from flask import Flask, jsonify, request
import logging

from mlp import MLPModel
from db_manager import DBManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_proc(epochs, eval, dl_id, dl_name):
    model = MLPModel()
    model.train(epochs, eval, dl_id, dl_name)


def serv_proc(dl_id, dl_name):
    model = MLPModel()
    model.serv(dl_id, dl_name)
    pass


train_process = None
serv_process = None


def get_api_server(db_client):
    app = Flask(__name__)

    version = 'v1.0'

    info = {
        'name': 'rmos prediction api',
        'version': version
    }

    def stop_training():
        global train_process

        if train_process is not None and train_process.is_alive():
            train_process.terminate()

    def stop_serving():
        global serv_process

        if serv_process is not None and serv_process.is_alive():
            serv_process.terminate()

    @app.route("/reset", methods=['POST', 'GET'])
    def reset():
        stop_training()

        check_point = "./model_weight"

        if os.path.exists(check_point):
            shutil.rmtree(check_point)

        response = {
            'status': 1,
            'message': 'reset successful'
        }

        return jsonify(response)

    @app.route("/stop", methods=['POST', 'GET'])
    def stop():
        stop_training()
        stop_serving()

    @app.route("/train", methods=['POST'])
    def train():
        logging.debug("train request")

        epochs = request.json['EPOCHS']
        eval = request.json['EVAL']
        dl_id = request.json['DL_ID']
        dl_name = request.json['DL_NAME']

        global train_process

        try:
            stop_training()
            train_process = Process(target=train_proc, args=[int(epochs), int(eval), int(dl_id), dl_name])
            train_process.start()

            response = {
                'status': 1,
                'message': 'train started'
            }

        except Exception as e:
            response = {
                'status': 0,
                'message': e
            }

        return jsonify(response)

    @app.route("/train/progress", methods=['POST'])
    def train_progress():
        logging.debug("train progress request")
        response = MLPModel.load_status()

        if response is None:
            response = {
                'epoch': 0,
                'loss': str(0),
                'acc': str(0)
            }
        else:
            logging.debug(response['epoch'])
            logging.debug(response['loss'])
            logging.debug(response['acc'])
        return jsonify(response)

    @app.route("/serv", methods=['POST'])
    def serv():
        print('serv')
        logging.debug("serv request")

        dl_id = request.json['DL_ID']
        dl_name = request.json['DL_NAME']

        global serv_process

        try:
            stop_serving()
            serv_process = Process(target=serv_proc, args=[int(dl_id), dl_name])
            serv_process.start()

            response = {
                'status': 1,
                'message': 'serv started'
            }

        except Exception as e:
            response = {
                'status': 0,
                'message': e
            }

        return jsonify(response)

    @app.route("/serv/progress")
    def serv_progress():
        logging.debug("serv progress request")
        return "serv progress"

    return app


def main():
    # logging.basicConfig(filename='rmos.log', level=logging.DEBUG)

    model_save_path = path.join(path.dirname(path.abspath(__file__)), 'model_save')

    api_server = get_api_server(db_client=DBManager())
    api_server.run(host='0.0.0.0', debug=False)


if __name__ == '__main__':
    main()
