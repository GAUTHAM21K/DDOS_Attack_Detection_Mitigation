from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from ryu.lib.packet import packet
from ryu.lib.packet.stream_parser import StreamParser, TooSmallException

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end-start))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
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
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        with open("PredictFlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            for stat in sorted([flow for flow in ev.msg.body if flow.priority == 1],
                               key=lambda flow: (flow.match.get('eth_type'), flow.match.get('ipv4_src'), flow.match.get('ipv4_dst'), flow.match.get('ip_proto'))):
                try:
                    ip_src = stat.match.get('ipv4_src', '0.0.0.0')
                    ip_dst = stat.match.get('ipv4_dst', '0.0.0.0')
                    ip_proto = stat.match.get('ip_proto', 0)
                    icmp_code = stat.match.get('icmpv4_code', -1)
                    icmp_type = stat.match.get('icmpv4_type', -1)
                    tp_src = stat.match.get('tcp_src') or stat.match.get('udp_src') or 0
                    tp_dst = stat.match.get('tcp_dst') or stat.match.get('udp_dst') or 0
                    flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"
                    packet_count_per_second = stat.packet_count / stat.duration_sec if stat.duration_sec else 0
                    packet_count_per_nsecond = stat.packet_count / stat.duration_nsec if stat.duration_nsec else 0
                    byte_count_per_second = stat.byte_count / stat.duration_sec if stat.duration_sec else 0
                    byte_count_per_nsecond = stat.byte_count / stat.duration_nsec if stat.duration_nsec else 0
                    file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{packet_count_per_second},{packet_count_per_nsecond},{byte_count_per_second},{byte_count_per_nsecond}\n")
                except TooSmallException:
                    continue

    def flow_training(self):
        self.logger.info("Flow Training ...")
        model_path = "knn_model.pkl"
        scaler_path = "scaler.pkl"
        selected_features = ['timestamp', 'ip_src', 'tp_src', 'ip_dst', 'tp_dst', 'ip_proto',
                             'icmp_code', 'icmp_type', 'flow_duration_sec', 'flow_duration_nsec',
                             'idle_timeout', 'hard_timeout', 'flags', 'packet_count', 'byte_count',
                             'packet_count_per_second', 'packet_count_per_nsecond',
                             'byte_count_per_second', 'byte_count_per_nsecond']

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.flow_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Loaded trained model from disk.")
        else:
            flow_dataset = pd.read_csv('FlowStatsfile.csv')
            flow_dataset['ip_src'] = flow_dataset['ip_src'].str.replace('.', '')
            flow_dataset['ip_dst'] = flow_dataset['ip_dst'].str.replace('.', '')
            flow_dataset['tp_src'] = flow_dataset['tp_src'].astype(str)
            flow_dataset['tp_dst'] = flow_dataset['tp_dst'].astype(str)

            X_flow = flow_dataset[selected_features].values.astype('float64')
            self.scaler = StandardScaler().fit(X_flow)
            X_scaled = self.scaler.transform(X_flow)

            y_flow = flow_dataset.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_flow, test_size=0.25, random_state=0)
            classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            self.flow_model = classifier.fit(X_train, y_train)
            joblib.dump(self.flow_model, model_path)
            joblib.dump(self.scaler, scaler_path)

            y_pred = self.flow_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info("confusion matrix")
            self.logger.info(confusion_matrix(y_test, y_pred))
            self.logger.info("success accuracy = {0:.2f} %".format(acc*100))
            self.logger.info("fail accuracy = {0:.2f} %".format((1.0 - acc)*100))
            self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            selected_features = ['timestamp', 'ip_src', 'tp_src', 'ip_dst', 'tp_dst', 'ip_proto',
                                 'icmp_code', 'icmp_type', 'flow_duration_sec', 'flow_duration_nsec',
                                 'idle_timeout', 'hard_timeout', 'flags', 'packet_count', 'byte_count',
                                 'packet_count_per_second', 'packet_count_per_nsecond',
                                 'byte_count_per_second', 'byte_count_per_nsecond']

            predict_df = pd.read_csv('PredictFlowStatsfile.csv')
            if predict_df.empty:
                self.logger.warning("No flow stats available for prediction.")
                return

            predict_df['ip_src'] = predict_df['ip_src'].str.replace('.', '')
            predict_df['ip_dst'] = predict_df['ip_dst'].str.replace('.', '')
            predict_df['tp_src'] = predict_df['tp_src'].astype(str)
            predict_df['tp_dst'] = predict_df['tp_dst'].astype(str)

            X_predict = predict_df[selected_features].values.astype('float64')
            X_scaled = self.scaler.transform(X_predict)

            y_pred = self.flow_model.predict(X_scaled)
            legitimate, ddos = 0, 0

            for i, label in enumerate(y_pred):
                if label == 0:
                    legitimate += 1
                else:
                    ddos += 1
                    dpid = int(predict_df.iloc[i, 1])
                    victim_ip = predict_df.iloc[i, 5]
                    attacked_port = int(predict_df.iloc[i, 6])
                    self.mitigate_attack(dpid, victim_ip, attacked_port)

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate / len(y_pred)) * 100 > 80:
                self.logger.info("Legitimate traffic detected.")
            else:
                self.logger.info("DDoS traffic detected and blocked.")
            self.logger.info("------------------------------------------------------------------------------")

            open("PredictFlowStatsfile.csv", "w").write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")

    def mitigate_attack(self, datapath_id, victim_ip, port_no):
        self.logger.info(f"Blocking {victim_ip}:{port_no} on switch {datapath_id}")
        dp = self.datapaths.get(datapath_id)
        if not dp:
            self.logger.warning(f"Datapath {datapath_id} not found")
            return

        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=victim_ip, tcp_dst=port_no)
        actions = []
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=200, match=match, instructions=inst)
        dp.send_msg(mod)
        self.logger.info(f"Blocked {victim_ip}:{port_no} on switch {datapath_id}")
