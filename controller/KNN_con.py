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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.scaler = None
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
            for stat in sorted([flow for flow in ev.msg.body if flow.priority == 1], key=lambda flow:
                    (flow.match.get('eth_type'), flow.match.get('ipv4_src'), flow.match.get('ipv4_dst'), flow.match.get('ip_proto'))):

                ip_src = stat.match.get('ipv4_src', '0.0.0.0')
                ip_dst = stat.match.get('ipv4_dst', '0.0.0.0')
                ip_proto = stat.match.get('ip_proto', 0)
                tp_src = stat.match.get('tcp_src', stat.match.get('udp_src', 0))
                tp_dst = stat.match.get('tcp_dst', stat.match.get('udp_dst', 0))
                icmp_code = stat.match.get('icmpv4_code', -1)
                icmp_type = stat.match.get('icmpv4_type', -1)

                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"
                try:
                    pps = stat.packet_count / stat.duration_sec if stat.duration_sec > 0 else 0
                    ppns = stat.packet_count / stat.duration_nsec if stat.duration_nsec > 0 else 0
                    bps = stat.byte_count / stat.duration_sec if stat.duration_sec > 0 else 0
                    bpns = stat.byte_count / stat.duration_nsec if stat.duration_nsec > 0 else 0
                except ZeroDivisionError:
                    pps = ppns = bps = bpns = 0

                file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{pps},{ppns},{bps},{bpns}\n")

    def flow_training(self):
        self.logger.info("Flow Training ...")
        flow_dataset = pd.read_csv('FlowStatsfile.csv')

        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '', regex=False)
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '', regex=False)
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '', regex=False)

        X = flow_dataset.iloc[:, :-1].values.astype('float64')
        y = flow_dataset.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        classifier = KNeighborsClassifier(n_neighbors=5)
        self.flow_model = classifier.fit(X_train, y_train)

        joblib.dump(self.flow_model, 'knn_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')

        y_pred = self.flow_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info("Confusion matrix:\n%s", cm)
        self.logger.info("Success accuracy = %.2f %%", acc * 100)
        self.logger.info("Fail accuracy = %.2f %%", (1 - acc) * 100)
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            predict_df = pd.read_csv("PredictFlowStatsfile.csv")

            predict_df.iloc[:, 2] = predict_df.iloc[:, 2].astype(str).str.replace('.', '', regex=False)
            predict_df.iloc[:, 3] = predict_df.iloc[:, 3].astype(str).str.replace('.', '', regex=False)
            predict_df.iloc[:, 5] = predict_df.iloc[:, 5].astype(str).str.replace('.', '', regex=False)

            X = predict_df.iloc[:, :].values.astype('float64')

            if not hasattr(self, 'flow_model') or self.flow_model is None:
                self.flow_model = joblib.load('knn_model.pkl')
                self.scaler = joblib.load('scaler.pkl')

            X = self.scaler.transform(X)

            y_pred = self.flow_model.predict(X)

            legitimate = sum(label == 0 for label in y_pred)
            ddos = sum(label == 1 for label in y_pred)

            self.logger.info("------------------------------------------------------------------------------")
            if legitimate / len(y_pred) > 0.8:
                self.logger.info("Legitimate traffic ...")
            else:
                self.logger.info("DDoS traffic detected ...")
                for index, label in enumerate(y_pred):
                    if label == 1:
                        victim = int(predict_df.iloc[index, 5]) % 20
                        ip_src = predict_df.iloc[index, 3]
                        datapath_id = int(predict_df.iloc[index, 1])
                        self.logger.info("Blocking attacker with source IP: %s", ip_src)
                        self.block_attacker(datapath_id, ip_src)
                self.logger.info("Victim may be host: h%d", victim)

            self.logger.info("------------------------------------------------------------------------------")

            with open("PredictFlowStatsfile.csv", "w") as f:
                f.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")

    def block_attacker(self, datapath_id, ip_src):
        if datapath_id not in self.datapaths:
            self.logger.warning("Datapath %s not found", datapath_id)
            return

        datapath = self.datapaths[datapath_id]
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=ip_src)
        actions = []  # drop
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(datapath=datapath, priority=100, match=match,
                                instructions=inst, command=ofproto.OFPFC_ADD)
        datapath.send_msg(mod)
        self.logger.info("Drop rule installed for %s on datapath %s", ip_src, datapath_id)
