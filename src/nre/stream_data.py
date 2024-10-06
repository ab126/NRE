import pickle
import json
import pandas as pd
import numpy as np
import time

import logging
from websocket_server import WebsocketServer

from .network_connectivity import get_all_entities
from .preprocess import preprocess_df
from .real_time_model import NetworkModel
from .safe_routing import communication_graph_from_df
from .time_windowed import get_window


# Start Server
def start_web_socket_server():
    """ Start Websocket Server for sending data via amqp protocol """
    ws = WebsocketServer(host='127.0.0.1', port=15674, loglevel=logging.INFO)
    print("Starting AMQP Server")
    ws.run_forever(threaded=True)
    return ws


def start_stream(ws, df_conn, entity_names, window_type='conn', grow_entities=False, src_id_col=' Source IP',
                 dst_id_col=' Destination IP', graph_conn_size=150, conn_size=3, t_graph=4, t_sync=0.5,
                 forget_factor=0.5, relief_factor=0.7, display_time=3):
    """
    Sets up the server and starts streaming data

    :param ws: Web Socket Server
    :param df_conn: Connection data dataframe
    :param entity_names: Target Set of Entities
    :param window_type:
    :param grow_entities:
    :param src_id_col:
    :param dst_id_col:
    :param graph_conn_size:
    :param conn_size:
    :param t_graph:
    :param t_sync:
    :param forget_factor:
    :param relief_factor:
    :param display_time: Amount of time the graph is displayed on the rendering end
    :return:
    """
    n_entities = len(entity_names)

    for _ in range(1000):  # For x runs

        if grow_entities:
            nm = NetworkModel()
        else:
            nm = NetworkModel(entity_names=entity_names, mat_x_init=np.ones(n_entities), mat_p_init=np.eye(n_entities))

        date_col = ' Timestamp'
        end_of_df = False
        start_of_df = True
        current_datetime = df_conn.iloc[0][date_col]
        last_datetime = df_conn.iloc[-1][date_col]
        i = 0

        print("Waiting for Client")

        # Await until connection
        while len(ws.clients) == 0:
            time.sleep(1)
            continue

        print("Client Connected")

        edges_list = []
        ind = 0
        n_flows = 0
        n_new = 0
        while end_of_df is False:

            if start_of_df and not grow_entities:  # Initial Run
                start_of_df = False
                time.sleep(display_time/2)
                continue
            else:
                if window_type == 'time':
                    temp_df, current_datetime = get_window(current_datetime, df_conn, date_col=date_col,
                                                           time_window=t_graph,
                                                           time_scale='sec')
                else:  # 'conn'
                    temp_df, current_datetime = df_conn.iloc[ind:ind + graph_conn_size, :], \
                        df_conn.iloc[ind + graph_conn_size][date_col]
                    ind += graph_conn_size

                if current_datetime >= last_datetime:
                    end_of_df = True
                # if temp_df.empty or len(temp_df.shape) < 2 or temp_df.shape[0] < MIN_SAMPLES:
                #    continue

                i += 1
                print("Graph #" + str(i))

                if grow_entities:
                    curr_entities = get_all_entities(temp_df, src_id_col=src_id_col, dst_id_col=dst_id_col)
                    n_new = len([name for name in curr_entities if name not in nm.entity_names])
                    nm.add_entities(curr_entities)
                # try:
                nm.update_new_tick_conn_data(temp_df, conn_param='NPR', sync_window_size=t_sync,
                                             window_type=window_type, conn_size=conn_size, keep_unit=True,
                                             forget_factor=forget_factor,
                                             relief_factor=relief_factor)
                # except AssertionError:
                # print(AssertionError)
                # break
                nm.normalize_risks()

                g_topo = communication_graph_from_df(temp_df, entity_names=nm.entity_names, keep_outsiders=False)
                edges_list = list(g_topo.edges)
                n_flows = temp_df.shape[0]

            # Send the results
            if grow_entities:
                json_string = json.dumps({'names': nm.entity_names, 'funcEdges': nm.mat_f.tolist(),
                                          'riskArr': nm.mat_x.tolist(), 'riskCov': nm.mat_p.tolist(),
                                          'nFlows': [n_flows], 'timeStamp': [current_datetime.strftime('%X')],
                                          'topologyEdges': edges_list, 'newEntities': n_new})
            else:
                json_string = json.dumps({'names': nm.entity_names, 'funcEdges': nm.mat_f.tolist(),
                                          'riskArr': nm.mat_x.tolist(), 'riskCov': nm.mat_p.tolist(),
                                          'nFlows': [n_flows], 'timeStamp': [current_datetime.strftime('%X')],
                                          'topologyEdges': edges_list, 'newEntities': n_new})

            #
            try:
                ws.send_message(ws.clients[0], json_string)
            except:
                break
            # server.send_message_to_all(jsonstring)
            time.sleep(display_time)


if __name__ == '__main__':
    # Read Data
    df_raw = pd.read_csv(
        '../../CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv',
        header=0, encoding='cp1252')
    df = preprocess_df(df_raw, date_col=' Timestamp')

    df_temp = df.iloc[:10000, :]

    with open(r'../../tests/saves/victim_net.pickle', 'rb') as handle:
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
