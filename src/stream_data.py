import pickle
import json
import pandas as pd
import numpy as np
import time

import logging
from websocket_server import WebsocketServer

from src.preprocess import preprocess_df
from src.real_time_model import NetworkModel
from src.safe_routing import communication_graph_from_df
from src.time_windowed import get_window


# Start Server
def start_web_socket_server():
    """ Start Websocket Server for sending data via amqp protocol """
    ws = WebsocketServer(host='127.0.0.1', port=15674, loglevel=logging.INFO)
    print("Starting AMQP Server")
    ws.run_forever(threaded=True)
    return ws


# Sets up the server and starts streaming data
def start_stream(ws, df_conn, entity_names, window_type='conn'):

    n_entities = len(entity_names)
    t_sync, t_graph = .5, 4  # sec 2,5 00
    ff = 0.5
    rf = 0.7
    display_time = 7  # Amount of time the graph is displayed on the rendering end
    conn_size, graph_conn_size = 3, 15

    for _ in range(1000):  # For x runs

        nm = NetworkModel(entity_names=entity_names, mat_x_init=np.ones(n_entities), mat_p_init=np.eye(n_entities))
        date_col = ' Timestamp'
        end_of_df = False
        current_datetime = df_conn.iloc[0][date_col]
        last_datetime = df_conn.iloc[-1][date_col]
        i = 0

        print("Waiting for Client")

        # Await until connection
        while len(ws.clients) == 0:
            time.sleep(1)
            continue

        print("Client Connected")
        jsonstring = json.dumps(
            {'funcEdges': nm.mat_f.tolist(), 'risk_mean': {name: x for name, x in zip(nm.entity_names, nm.mat_x)},
             'risk_cov': nm.mat_p.tolist(), 'nFlows': [0], 'timeStamp': [current_datetime.strftime('%X')],
             'topologyEdges': []})
        ws.send_message(ws.clients[0], jsonstring)
        time.sleep(display_time)

        ind = 0
        while end_of_df is False:
            if window_type == 'time':
                temp_df, current_datetime = get_window(current_datetime, df_conn, date_col=date_col, time_window=t_graph,
                                                       time_scale='sec')
            else:  # 'conn'
                temp_df, current_datetime = df_conn.iloc[ind:ind+graph_conn_size, :], df_conn.iloc[ind+graph_conn_size][date_col]
                ind += graph_conn_size

            if current_datetime >= last_datetime:
                end_of_df = True
            # if temp_df.empty or len(temp_df.shape) < 2 or temp_df.shape[0] < MIN_SAMPLES:
            #    continue

            i += 1
            print("Graph #" + str(i))
            print(temp_df)

            # try:
            nm.update_new_tick(temp_df, conn_param='NPR', sync_window_size=t_sync, window_type=window_type,
                               conn_size=conn_size, keep_unit=True, forget_factor=ff, relief_factor=rf)
            # except AssertionError:
            # print(AssertionError)
            # break
            mat_x, mat_p, mat_f, window_names = nm.mat_x, nm.mat_p, nm.mat_f, nm.entity_names

            scale = mat_x.max() / 2
            mat_x = mat_x / scale
            mat_p = mat_p / scale ** 2

            # Send the results
            g_topo = communication_graph_from_df(temp_df, entity_names=entity_names, keep_outsiders=False)
            edges_list = list(g_topo.edges)
            jsonstring = json.dumps(
                {'funcEdges': mat_f.tolist(), 'risk_mean': {name: x for name, x in zip(window_names, mat_x)},
                 'risk_cov': mat_p.tolist(), 'nFlows': [temp_df.shape[0]],
                 'timeStamp': [current_datetime.strftime('%X')],
                 'topologyEdges': edges_list})

            #
            try:
                ws.send_message(ws.clients[0], jsonstring)
            except:
                break
            # server.send_message_to_all(jsonstring)
            time.sleep(display_time)


if __name__ == '__main__':
    # Read Data
    df_raw = pd.read_csv('../CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv',
                         header=0, encoding='cp1252')
    df = preprocess_df(df_raw, date_col=' Timestamp')

    df_temp = df.iloc[:10000, :]

    with open(r'../tests/saves/victim_net.pickle', 'rb') as handle:
        names = pickle.load(handle)

    server = start_web_socket_server()
    start_stream(server, df, names)

    """
    queue = queue.Queue()
    j0 = Thread(target=start_server, args=(queue,))
    j0.start()

    j1 = Thread(target=graph_process, args=(df_temp, entity_names, queue,))

    j0.join()
    j1.join()
    """
