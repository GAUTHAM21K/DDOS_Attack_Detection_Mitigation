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

class IntegratedMonitor(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(IntegratedMonitor, self).__init__(*args, **kwargs)
        self.datapaths = {}  # Dictionary to track registered datapaths
        self.monitor_thread = hub.spawn(self._monitor)  # Thread for periodic monitoring
        self.mitigation = 0  # Indicator for ongoing mitigation

        # Train the model
        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

    # Handler for datapath state changes
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

    # Periodic monitoring of flow stats
    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)  # Pause 10 seconds before the next monitoring cycle
            self.flow_predict()

    # Request flow statistics from a datapath
    def _request_stats(self, datapath):
        self.logger.debug('Send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # Handler for incoming flow stats and writing them to the file
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        # Collect flow stats and write them to PredictFlowStatsfile.csv
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
                   
