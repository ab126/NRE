import pickle
import json
import pandas as pd
import numpy as np
import time

import logging
from websocket_server import WebsocketServer

from src.preprocess import preprocess_df
from src.real_time_model import NetworkModel
from src.time_windowed import get_window


# Start Server
def start_server():
    """ Start Websocket Server for sending data via amqp protocol """
    server = WebsocketServer(host='127.0.0.1', port=15674, loglevel=logging.INFO)
    print("Starting AMQP Server")
    server.run_forever(threaded=True)
    return server


# Start Network Model

def graph_process(df, entity_names, server):
    """ Process for calculating single graph and sending the updates """

    n_entities = len(entity_names)
    t_sync, t_graph = 20, 5000  # sec
    nm = NetworkModel(entity_names=entity_names, mat_x_init=np.ones(n_entities), mat_p_init=np.eye(n_entities))

    date_col = ' Timestamp'

    end_of_df = False
    i = 0
    current_datetime = df.iloc[0][date_col]
    last_datetime = df.iloc[-1][date_col]

    print("Waiting for Client")

    # Await until connection
    while len(server.clients) == 0:
        time.sleep(1)
        continue

    print("Client Connected")

    while end_of_df is False:
        temp_df, current_datetime = get_window(current_datetime, df, date_col=date_col, time_window=t_graph,
                                               time_scale='sec')
        if current_datetime >= last_datetime:
            end_of_df = True

        i += 1
        print("Graph #" + str(i))

        nm.update_new_tick(temp_df, conn_param='NPR', sync_window_size=t_sync, time_scale='sec', keep_unit=True)
        mat_x, mat_p, mat_f, names = nm.mat_x, nm.mat_p, nm.cu.mat_f, nm.cu.names

        # Send the results
        jsonstring = json.dumps({'funcEdges': mat_f.tolist(), 'risk_mean': {name: x for name, x in zip(names, mat_x)},
                                 'risk_cov': mat_p.tolist()})

        server.send_message(server.clients[0], jsonstring)
        time.sleep(1)


if __name__ == '__main__':
    # Read Data
    df_raw = pd.read_csv('..\CIC-IDS-2017\GeneratedLabelledFlows\TrafficLabelling\Tuesday-WorkingHours.pcap_ISCX.csv',
                         header=0, encoding='cp1252')
    df = preprocess_df(df_raw, date_col=' Timestamp')

    df_temp = df.iloc[:10000, :]

    with open(r'saves\victim_net.pickle', 'rb') as handle:
        entity_names = pickle.load(handle)

    server = start_server()
    graph_process(df_temp, entity_names, server)

    """
    queue = queue.Queue()
    j0 = Thread(target=start_server, args=(queue,))
    j0.start()

    j1 = Thread(target=graph_process, args=(df_temp, entity_names, queue,))

    j0.join()
    j1.join()
    """
