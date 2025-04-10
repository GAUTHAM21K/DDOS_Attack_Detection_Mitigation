from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.mitigation = 0  # Flag for mitigation

        # Train the Random Forest model
        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('Register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('Unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)  # Pause for 10 seconds before the next monitoring cycle
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('Send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        # Collect and write incoming flow stats to PredictFlowStatsfile.csv
        timestamp = datetime.now().timestamp()
        body = ev.msg.body

        # Open file for writing data
        with open("PredictFlowStatsfile.csv", "a") as file:
            for stat in sorted([flow for flow in body if flow.priority == 1], key=lambda flow:
                               (flow.match.get('ipv4_src', ''), flow.match.get('ipv4_dst', ''))):
                # Extract flow attributes
                ip_src = stat.match.get('ipv4_src', '0.0.0.0')
                ip_dst = stat.match.get('ipv4_dst', '0.0.0.0')
                ip_proto = stat.match.get('ip_proto', 0)
                tp_src = stat.match.get('tcp_src', 0) or stat.match.get('udp_src', 0)
                tp_dst = stat.match.get('tcp_dst', 0) or stat.match.get('udp_dst', 0)

                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"

                try:
                    packet_count_per_second = stat.packet_count / max(1, stat.duration_sec)
                    byte_count_per_second = stat.byte_count / max(1, stat.duration_sec)
                except ZeroDivisionError:
                    packet_count_per_second = 0
                    byte_count_per_second = 0

                # Write flow statistics to the file
                file.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},"
                           f"{stat.packet_count},{stat.byte_count},{packet_count_per_second},{byte_count_per_second}\n")

    def flow_training(self):
        self.logger.info("Flow Training ...")

        # Load and preprocess training dataset
        flow_dataset = pd.read_csv('FlowStatsfile.csv')
        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

        X_flow = flow_dataset.iloc[:, :-1].values.astype('float64')
        y_flow = flow_dataset.iloc[:, -1].values

        # Train/test split and train Random Forest model
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)
        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.flow_model = classifier.fit(X_flow_train, y_flow_train)

        # Evaluate model
        y_flow_pred = self.flow_model.predict(X_flow_test)
        self.logger.info("Confusion Matrix:")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)
        self.logger.info("Success Accuracy = {0:.2f}%".format(acc * 100))
        fail = 1.0 - acc
        self.logger.info("Fail Accuracy = {0:.2f}%".format(fail * 100))

    def flow_predict(self):
        try:
            # Load the CSV file
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            # Validate dataset
            if predict_flow_dataset.empty:
                self.logger.error("PredictFlowStatsfile.csv is empty or invalid.")
                return

            # Preprocess the dataset
            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            # Convert to NumPy array and predict
            X_predict_flow = predict_flow_dataset.iloc[:, :].values.astype('float64')
            y_flow_pred = self.flow_model.predict(X_predict_flow)

            # Analyze prediction results
            legitimate_traffic = sum(y_flow_pred == 0)
            ddos_traffic = sum(y_flow_pred == 1)

            self.logger.info("Traffic Analysis:")
            if (legitimate_traffic / len(y_flow_pred) * 100) > 80:
                self.logger.info("Traffic is Legitimate!")
            else:
                self.logger.info("NOTICE!! DoS Attack in Progress!!!")
                victim_index = y_flow_pred.tolist().index(1)  # Find the first occurrence of malicious traffic
                victim = int(predict_flow_dataset.iloc[victim_index, 5]) % 20
                self.logger.info("Victim Host: h{}".format(victim))
                self.mitigation = 1
                self.logger.info("Mitigation in Progress!")

        except Exception as e:
            # Log any errors during prediction
            self.logger.error("Error in flow prediction: {}".format(e))

        finally:
            # Reset the file for the next monitoring cycle
            with open("PredictFlowStatsfile.csv", "w") as file0:
                file0.write(
                    'timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,'
                    'flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,'
                    'packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n'
                )
