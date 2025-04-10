from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3

import switch
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        # Dictionary to keep track of blocked attackers
        self.blocked_attackers = {}

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end-start))

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
        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        file0 = open("PredictFlowStatsfile.csv","w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        for stat in sorted([flow for flow in body if (flow.priority == 1) ], key=lambda flow:
            (flow.match['eth_type'],flow.match['ipv4_src'],flow.match['ipv4_dst'],flow.match['ip_proto'])):
        
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            
            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
                
            elif stat.match['ip_proto'] == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']

            elif stat.match['ip_proto'] == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
          
            try:
                packet_count_per_second = stat.packet_count/stat.duration_sec
                packet_count_per_nsecond = stat.packet_count/stat.duration_nsec
            except:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0
                
            try:
                byte_count_per_second = stat.byte_count/stat.duration_sec
                byte_count_per_nsecond = stat.byte_count/stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0
                
            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src,ip_dst, tp_dst,
                        stat.match['ip_proto'],icmp_code,icmp_type,
                        stat.duration_sec, stat.duration_nsec,
                        stat.idle_timeout, stat.hard_timeout,
                        stat.flags, stat.packet_count,stat.byte_count,
                        packet_count_per_second,packet_count_per_nsecond,
                        byte_count_per_second,byte_count_per_nsecond))
            
        file0.close()

    def flow_training(self):
        self.logger.info("Flow Training ...")

        flow_dataset = pd.read_csv('FlowStatsfile.csv')

        flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].str.replace('.', '')
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')

        X_flow = flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = flow_dataset.iloc[:, -1].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = self.flow_model.predict(X_flow_test)

        self.logger.info("------------------------------------------------------------------------------")

        self.logger.info("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        self.logger.info("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype('float64')
            
            y_flow_pred = self.flow_model.predict(X_predict_flow)

            legitimate_traffic = 0
            ddos_traffic = 0
            attackers = set()
            victim_ip = None

            # Analyze prediction results and collect attacker IPs
            for i, prediction in enumerate(y_flow_pred):
                if prediction == 0:
                    legitimate_traffic += 1
                else:
                    ddos_traffic += 1
                    # Identify the victim and attacker
                    victim_ip = predict_flow_dataset.iloc[i, 5]  # IP destination
                    attacker_ip = predict_flow_dataset.iloc[i, 3]  # IP source
                    attackers.add(attacker_ip)

            self.logger.info("------------------------------------------------------------------------------")
            
            if (legitimate_traffic/len(y_flow_pred)*100) > 80:
                self.logger.info("Legitimate traffic detected...")
            else:
                # DDoS traffic detected - implement mitigation
                self.logger.info("DDoS traffic detected!")
                if victim_ip:
                    victim = int(float(victim_ip))%20
                    self.logger.info("Victim is host: h{}".format(victim))
                    self.logger.info("Implementing mitigation measures...")
                    
                    # Block all attackers
                    for attacker_ip in attackers:
                        self.block_attacker(attacker_ip, victim_ip)
                
            self.logger.info("------------------------------------------------------------------------------")
            
            file0 = open("PredictFlowStatsfile.csv","w")
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            file0.close()

        except Exception as e:
            self.logger.error(f"Error in flow prediction: {e}")

    def block_attacker(self, attacker_ip, victim_ip):
        """
        Add flow rules to block traffic from attacker to victim
        """
        # Skip if this attacker is already blocked
        if attacker_ip in self.blocked_attackers:
            self.logger.info(f"Attacker {attacker_ip} already blocked")
            return
        
        # Mark attacker as blocked with timestamp
        self.blocked_attackers[attacker_ip] = datetime.now()
        self.logger.info(f"Blocking attacker: {attacker_ip}")
        
        # Add drop rules to all datapaths
        for dp in self.datapaths.values():
            parser = dp.ofproto_parser
            ofproto = dp.ofproto
            
            # Create a match for the attacker's IP
            match = parser.OFPMatch(
                eth_type=0x0800,  # IPv4
                ipv4_src=attacker_ip
            )
            
            # Install flow entry with highest priority (100) and drop action
            actions = []  # Empty action list means drop
            self.add_flow(dp, 100, match, actions, idle_timeout=300)  # Block for 5 minutes
            
            self.logger.info(f"Installed drop rule for {attacker_ip} on datapath {dp.id}")
    
    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        """
        Add a flow entry to the datapath with specified parameters
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout
        )
        
        datapath.send_msg(mod)
        
    def remove_stale_blocks(self):
        """
        Remove block rules that have been in place for too long
        """
        now = datetime.now()
        expired_attackers = []
        
        # Find expired blocks (older than 5 minutes)
        for attacker_ip, block_time in self.blocked_attackers.items():
            if (now - block_time).total_seconds() > 300:  # 5 minutes
                expired_attackers.append(attacker_ip)
                
        # Remove expired blocks
        for attacker_ip in expired_attackers:
            self.logger.info(f"Removing block for {attacker_ip} - block duration expired")
            del self.blocked_attackers[attacker_ip]
            
            # Implement removing the flow rule if needed
            # (This would require tracking the rule IDs or using a special cookie value)
