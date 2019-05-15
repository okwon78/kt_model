from multiprocessing import Process
from os import path
from flask import Flask
import logging

from mlp import MLPModel
from db_manager import DBManager


def train_proc(epochs, eval):
    model = MLPModel()
    model.train(epochs, eval)


def serv_proc():
    model = MLPModel()
    # model.serv()
    pass


def get_api_server(db_client):
    app = Flask(__name__)

    version = 'v1.0'

    info = {
        'name': 'rmos prediction api',
        'version': version
    }

    train_proc = None
    serv_proc = None

    @app.route("/train", methods=['POST'])
    def train():
        logging.debug("train request")

        if train_proc is not None:
            train_proc.stop()
        process = Process(target=train_proc, args=(1000, 100))
        process.start()

        return "train"

    @app.route("/train/progress")
    def train_progress():
        logging.debug("train progress request")
        return "train progress"

    @app.route("/serv")
    def serv():

        logging.debug("serv request")

        if serv_proc is not None:
            serv_proc.stop()
        process = Process(target=serv_proc, args=(1000, 100))
        process.start()

        return "serv"

    @app.route("/serv/progress")
    def serv_progress():
        logging.debug("serv progress request")
        return "serv progress"

    return app


def main():
    # logging.basicConfig(filename='rmos.log', level=logging.DEBUG)

    model_save_path = path.join(path.dirname(path.abspath(__file__)), 'model_save')

    api_server = get_api_server(db_client=DBManager())
    api_server.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    main()
