from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor


class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            self._request_all_stats()
            self.flow_predict()
            hub.sleep(5)  # Reduced sleep interval for faster monitoring

    def _request_all_stats(self):
        with ThreadPoolExecutor() as executor:
            executor.map(self._request_stats, self.datapaths.values())

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        file0 = open("PredictFlowStatsfile.csv", "w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,'
                    'icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,'
                    'idle_timeout,hard_timeout,flags,packet_count,byte_count,'
                    'packet_count_per_second,packet_count_per_nsecond,'
                    'byte_count_per_second,byte_count_per_nsecond\n')

        body = ev.msg.body
        for stat in sorted([flow for flow in body if flow.priority == 1], key=lambda flow:
                           (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

            ip_src = stat.match.get('ipv4_src', '')
            ip_dst = stat.match.get('ipv4_dst', '')
            ip_proto = stat.match.get('ip_proto', 0)
            icmp_code = stat.match.get('icmpv4_code', -1)
            icmp_type = stat.match.get('icmpv4_type', -1)
            tp_src = stat.match.get('tcp_src', 0) or stat.match.get('udp_src', 0)
            tp_dst = stat.match.get('tcp_dst', 0) or stat.match.get('udp_dst', 0)

            flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"
            try:
                packet_count_per_second = stat.packet_count / stat.duration_sec
                packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
            except ZeroDivisionError:
                packet_count_per_second = packet_count_per_nsecond = 0

            try:
                byte_count_per_second = stat.byte_count / stat.duration_sec
                byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
            except ZeroDivisionError:
                byte_count_per_second = byte_count_per_nsecond = 0

            file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},"
                        f"{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},"
                        f"{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},"
                        f"{stat.byte_count},{packet_count_per_second},{packet_count_per_nsecond},"
                        f"{byte_count_per_second},{byte_count_per_nsecond}\n")
        file0.close()

    def flow_training(self):
        self.logger.info("Flow Training ...")

        flow_dataset = pd.read_csv('FlowStatsfile.csv')
        flow_dataset.replace({'\\.': ''}, regex=True, inplace=True)

        X_flow = flow_dataset.iloc[:, :-1].values.astype('float64')
        y_flow = flow_dataset.iloc[:, -1].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = self.flow_model.predict(X_flow_test)

        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info("Confusion Matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)
        self.logger.info(f"Success Accuracy = {acc*100:.2f} %")
        fail = 1.0 - acc
        self.logger.info(f"Fail Accuracy = {fail*100:.2f} %")
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
            predict_flow_dataset.replace({'\\.': ''}, regex=True, inplace=True)

            X_predict_flow = predict_flow_dataset.values.astype('float64')
            y_flow_pred = self.flow_model.predict(X_predict_flow)

            legitimate_trafic = (y_flow_pred == 0).sum()
            ddos_trafic = (y_flow_pred == 1).sum()

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_trafic / len(y_flow_pred) * 100) > 80:
                self.logger.info("Legitimate Traffic ...")
            else:
                self.logger.info("DDoS Traffic Detected ...")
            self.logger.info("------------------------------------------------------------------------------")

            predict_flow_dataset.iloc[0:0] = []
            predict_flow_dataset.to_csv("PredictFlowStatsfile.csv", index=False)

        except Exception as e:
            self.logger.error(f"Error in flow prediction: {e}")
