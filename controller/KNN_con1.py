from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3

import switch
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.flow_stats_buffer = []  # Buffer to collect stats before processing
        self.flow_model = None
        self.blocked_sources = set()  # Track blocked source IPs
        
        # Train the model immediately at startup
        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end-start))
        
        # Start monitoring thread after model is trained
        self.monitor_thread = hub.spawn(self._monitor)

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
            # Clear stats buffer before requesting new stats
            self.flow_stats_buffer = []
            
            # Request stats from all datapaths
            for dp in self.datapaths.values():
                self._request_stats(dp)
            
            # Give time for stats to be collected
            hub.sleep(5)  # Reduced from 10 to 5 for faster processing
            
            # Only predict if we've collected some data
            if self.flow_stats_buffer:
                self.flow_predict()
            else:
                self.logger.debug('No flow stats collected, skipping prediction')

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        body = ev.msg.body
        
        for stat in sorted([flow for flow in body if (flow.priority == 1)], key=lambda flow:
            (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):
            
            # Default values
            icmp_code = -1
            icmp_type = -1
            tp_src = 0
            tp_dst = 0
            
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            
            if ip_proto == 1:  # ICMP
                icmp_code = stat.match.get('icmpv4_code', -1)
                icmp_type = stat.match.get('icmpv4_type', -1)
            elif ip_proto == 6:  # TCP
                tp_src = stat.match.get('tcp_src', 0)
                tp_dst = stat.match.get('tcp_dst', 0)
            elif ip_proto == 17:  # UDP
                tp_src = stat.match.get('udp_src', 0)
                tp_dst = stat.match.get('udp_dst', 0)

            flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}".replace('.', '')
            
            # Avoid division by zero
            duration_sec = stat.duration_sec if stat.duration_sec > 0 else 1
            duration_nsec = stat.duration_nsec if stat.duration_nsec > 0 else 1
            
            packet_count_per_second = stat.packet_count / duration_sec
            packet_count_per_nsecond = stat.packet_count / duration_nsec
            byte_count_per_second = stat.byte_count / duration_sec
            byte_count_per_nsecond = stat.byte_count / duration_nsec
            
            # Store in buffer instead of writing to file immediately
            self.flow_stats_buffer.append({
                'timestamp': timestamp,
                'datapath_id': ev.msg.datapath.id,
                'flow_id': flow_id,
                'ip_src': ip_src.replace('.', ''),
                'tp_src': tp_src,
                'ip_dst': ip_dst.replace('.', ''),
                'tp_dst': tp_dst,
                'ip_proto': ip_proto,
                'icmp_code': icmp_code,
                'icmp_type': icmp_type,
                'flow_duration_sec': duration_sec,
                'flow_duration_nsec': duration_nsec,
                'idle_timeout': stat.idle_timeout,
                'hard_timeout': stat.hard_timeout,
                'flags': stat.flags,
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'packet_count_per_second': packet_count_per_second,
                'packet_count_per_nsecond': packet_count_per_nsecond,
                'byte_count_per_second': byte_count_per_second,
                'byte_count_per_nsecond': byte_count_per_nsecond
            })

    def flow_training(self):
        self.logger.info("Flow Training ...")
        
        try:
            # Load training data
            flow_dataset = pd.read_csv('FlowStatsfile.csv')
            
            # Check if the dataset is empty
            if flow_dataset.empty:
                self.logger.error("Training dataset is empty!")
                return
                
            # Preprocess data - remove dots from IP addresses
            flow_dataset['flow_id'] = flow_dataset['flow_id'].astype(str).str.replace('.', '')
            flow_dataset['ip_src'] = flow_dataset['ip_src'].astype(str).str.replace('.', '')
            flow_dataset['ip_dst'] = flow_dataset['ip_dst'].astype(str).str.replace('.', '')
            
            # Convert to float once, not repeatedly
            X_flow = flow_dataset.iloc[:, :-1].values.astype('float64')
            y_flow = flow_dataset.iloc[:, -1].values
            
            # Split data and train model
            X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(
                X_flow, y_flow, test_size=0.25, random_state=0)
            
            # Train the model
            classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            self.flow_model = classifier.fit(X_flow_train, y_flow_train)
            
            # Evaluate the model
            y_flow_pred = self.flow_model.predict(X_flow_test)
            
            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info("Confusion matrix:")
            cm = confusion_matrix(y_flow_test, y_flow_pred)
            self.logger.info(cm)
            
            acc = accuracy_score(y_flow_test, y_flow_pred)
            self.logger.info("Success accuracy = {0:.2f} %".format(acc*100))
            fail = 1.0 - acc
            self.logger.info("Fail accuracy = {0:.2f} %".format(fail*100))
            self.logger.info("------------------------------------------------------------------------------")
            
        except FileNotFoundError:
            self.logger.error("Training file 'FlowStatsfile.csv' not found!")
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")

    def flow_predict(self):
        try:
            # Skip prediction if buffer is empty
            if not self.flow_stats_buffer:
                self.logger.info("No flow stats to predict")
                return
                
            # Convert buffer to DataFrame
            predict_flow_dataset = pd.DataFrame(self.flow_stats_buffer)
            
            # Skip prediction if DataFrame is empty after conversion
            if predict_flow_dataset.empty:
                self.logger.info("Empty dataset after conversion, skipping prediction")
                return
                
            # Convert to numpy array
            X_predict_flow = predict_flow_dataset.values.astype('float64')
            
            # Check if model exists
            if self.flow_model is None:
                self.logger.error("Model not trained yet!")
                return
                
            # Make predictions
            y_flow_pred = self.flow_model.predict(X_predict_flow)
            
            # Count legitimate vs. DDoS traffic
            legitimate_traffic = np.sum(y_flow_pred == 0)
            ddos_traffic = np.sum(y_flow_pred == 1)
            total_predictions = len(y_flow_pred)
            
            # Identify attackers and victims if DDoS attack detected
            attack_sources = {}
            victim_hosts = {}
            
            if ddos_traffic > 0:
                for i, pred in enumerate(y_flow_pred):
                    if pred == 1:  # If this flow is classified as DDoS
                        # Extract victim IP and attacker IP from the DataFrame
                        victim_ip = predict_flow_dataset.iloc[i]['ip_dst']
                        attacker_ip = predict_flow_dataset.iloc[i]['ip_src']
                        datapath_id = predict_flow_dataset.iloc[i]['datapath_id']
                        
                        # Get the last octet which usually identifies the host in a subnet
                        victim_id = int(victim_ip) % 100
                        attacker_id = int(attacker_ip) % 100
                        
                        # Count occurrences of each victim
                        if victim_id in victim_hosts:
                            victim_hosts[victim_id] += 1
                        else:
                            victim_hosts[victim_id] = 1
                            
                        # Count occurrences of each attacker and track their original IPs
                        attack_key = (datapath_id, attacker_ip)  # Include datapath ID to know which switch to program
                        if attack_key in attack_sources:
                            attack_sources[attack_key] += 1
                        else:
                            attack_sources[attack_key] = 1
            
            # Log results
            self.logger.info("------------------------------------------------------------------------------")
            legitimate_percentage = (legitimate_traffic / total_predictions) * 100
            self.logger.info(f"Legitimate traffic: {legitimate_traffic}/{total_predictions} ({legitimate_percentage:.2f}%)")
            self.logger.info(f"DDoS traffic: {ddos_traffic}/{total_predictions} ({100-legitimate_percentage:.2f}%)")
            
            if legitimate_percentage > 80:
                self.logger.info("Traffic analysis: LEGITIMATE TRAFFIC")
            else:
                self.logger.info("Traffic analysis: DDOS ATTACK DETECTED")
                
                # Report the most targeted victim
                if victim_hosts:
                    most_targeted = max(victim_hosts.items(), key=lambda x: x[1])
                    self.logger.info(f"Most targeted victim is host: h{most_targeted[0]} (targeted {most_targeted[1]} times)")
                
                # Identify and block top attackers
                if attack_sources:
                    # Sort attackers by number of malicious flows
                    sorted_attackers = sorted(attack_sources.items(), key=lambda x: x[1], reverse=True)
                    
                    # Block up to top 5 attackers (to avoid excessive blocking)
                    for (dp_id, attacker_ip), count in sorted_attackers[:5]:
                        # Convert numeric IP back to dotted format (if needed)
                        ip_parts = []
                        temp_ip = int(attacker_ip)
                        for _ in range(4):
                            ip_parts.insert(0, str(temp_ip % 256))
                            temp_ip //= 256
                        formatted_ip = '.'.join(ip_parts)
                        
                        # Only block if not already blocked
                        if attacker_ip not in self.blocked_sources:
                            self.block_attacker(dp_id, formatted_ip)
                            self.blocked_sources.add(attacker_ip)
                            self.logger.info(f"BLOCKED attacker with IP {formatted_ip} (attack count: {count})")
            
            self.logger.info("------------------------------------------------------------------------------")
            
            # Clear buffer after prediction
            self.flow_stats_buffer = []
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            # Don't clear buffer so we can debug if needed
    
    def block_attacker(self, datapath_id, ip_src):
        """
        Install a flow rule to block all traffic from the specified source IP
        """
        try:
            # Get the datapath object
            datapath = self.datapaths.get(datapath_id)
            if not datapath:
                self.logger.error(f"Datapath {datapath_id} not found!")
                return
                
            # Set up flow match and action
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            
            # Match all traffic from the attacker's IP
            match = parser.OFPMatch(
                eth_type=0x0800,  # IPv4
                ipv4_src=ip_src
            )
            
            # Empty action list means DROP
            actions = []
            
            # Create flow mod message
            inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            
            # Higher priority than regular flows to ensure blocking works
            priority = 100
            
            # Set a hard timeout of 300 seconds (5 minutes) to avoid permanent blocking
            # This can be adjusted based on security policy
            hard_timeout = 300
            
            # Create and send the flow modification message
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=inst,
                hard_timeout=hard_timeout,
                flags=ofproto.OFPFF_SEND_FLOW_REM  # Get notification when flow is removed
            )
            
            datapath.send_msg(mod)
            self.logger.info(f"Installed blocking rule for attacker {ip_src} on datapath {datapath_id}")
            
        except Exception as e:
            self.logger.error(f"Error installing blocking rule: {str(e)}")
    
    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        """
        Handle flow removal events (when blocked flows expire)
        """
        msg = ev.msg
        match = msg.match
        
        # Check if this was one of our blocking rules
        if match.get('eth_type') == 0x0800 and 'ipv4_src' in match:
            blocked_ip = match['ipv4_src']
            num_ip = int(blocked_ip.replace('.', ''))
            
            # Remove from our blocked set
            if num_ip in self.blocked_sources:
                self.blocked_sources.remove(num_ip)
                self.logger.info(f"Blocking rule for {blocked_ip} has expired or been removed")
