from __future__ import print_function
from flask import Flask, request, jsonify, json
import logging
from final_code_ipc_ import IPC_Check, IPC_Self_learning_Model, Improve
from ipc_detail import data_retrival
app = Flask(__name__)





@app.post("/ipc")
def get_ipc():
    corpus = request.json["corpus"]
    print(corpus)
    ipc = IPC_Self_learning_Model([corpus])
    detail, punishment = data_retrival(ipc)
    alldata = {'ipc': ipc, 'detail' :detail, 'punishment' : punishment }
    return alldata

@app.post("/approve")
def approve_ipc():
    res_ipc = request.json["ipc"]
    corpus = request.json["corpus"]
    pred_ipc = IPC_Self_learning_Model([corpus])
    Improve(IPC_Check(pred_ipc, res_ipc),corpus,res_ipc)
    return ""