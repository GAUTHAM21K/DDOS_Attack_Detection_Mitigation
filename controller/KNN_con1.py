from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import packet

import switch
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import logging
import sys

# Custom logger to filter out specific error messages
class ErrorFilter(logging.Filter):
    def filter(self, record):
        # Filter out the specific BGP error
        return "ryu.lib.packet.bgp" not in record.getMessage() and "120 < 22616" not in record.getMessage()

# Configure root logger with our filter
root_logger = logging.getLogger()
root_logger.addFilter(ErrorFilter())

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.flow_stats_buffer = []  # Buffer to collect stats before processing
        self.flow_model = None
        
        # Train the model immediately at startup
        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end-start))
        
        # Start monitoring thread after model is trained
        self.monitor_thread = hub.spawn(self._monitor)

    # We're NOT overriding packet_in_handler now - let it process normally
    # This ensures packet processing still works, we're just filtering errors

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
            
            # Give time for stats to be collected - not too short
            hub.sleep(5)  # 5 seconds should be sufficient for stats collection
            
            # Only predict if we've collected some data
            if self.flow_stats_buffer:
                self.flow_predict()
            else:
                self.logger.info('No flow stats collected, checking again soon')

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        body = ev.msg.body
        
        # Log how many flows we're processing to help with debugging
        self.logger.info(f"Processing {len(body)} flows from datapath {ev.msg.datapath.id}")
        
        flows_processed = 0
        for stat in sorted([flow for flow in body if (flow.priority == 1)], key=lambda flow:
            (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):
            
            try:
                # Default values
                icmp_code = -1
                icmp_type = -1
                tp_src = 0
                tp_dst = 0
                
                # Extract match fields carefully
                if 'ipv4_src' not in stat.match or 'ipv4_dst' not in stat.match or 'ip_proto' not in stat.match:
                    continue  # Skip if missing required fields
                
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

                # Remove dots from IP addresses in flow_id
                ip_src_clean = str(ip_src).replace('.', '')
                ip_dst_clean = str(ip_dst).replace('.', '')
                flow_id = f"{ip_src_clean}{tp_src}{ip_dst_clean}{tp_dst}{ip_proto}"
                
                # Avoid division by zero
                duration_sec = max(stat.duration_sec, 1)  # Ensure at least 1 second
                duration_nsec = max(stat.duration_nsec, 1)  # Ensure at least 1 nanosecond
                
                packet_count_per_second = stat.packet_count / duration_sec
                packet_count_per_nsecond = stat.packet_count / duration_nsec
                byte_count_per_second = stat.byte_count / duration_sec
                byte_count_per_nsecond = stat.byte_count / duration_nsec
                
                # Store in buffer instead of writing to file
                self.flow_stats_buffer.append({
                    'timestamp': timestamp,
                    'datapath_id': ev.msg.datapath.id,
                    'flow_id': flow_id,
                    'ip_src': ip_src_clean,
                    'tp_src': tp_src,
                    'ip_dst': ip_dst_clean,
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
                flows_processed += 1
                
            except Exception as e:
                # Log the exception but don't display it to console
                self.logger.debug(f"Error processing flow: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {flows_processed} flows")

    def flow_training(self):
        self.logger.info("Flow Training ...")
        
        try:
            # Load training data
            flow_dataset = pd.read_csv('FlowStatsfile.csv')
            
            # Check if the dataset is empty
            if flow_dataset.empty:
                self.logger.error("Training dataset is empty!")
                return
            
            self.logger.info(f"Loaded training dataset with {len(flow_dataset)} records")
                
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
            
            self.logger.info(f"Predicting on {len(predict_flow_dataset)} flow records")
                
            # Ensure all expected columns are present
            required_columns = [
                'timestamp', 'datapath_id', 'flow_id', 'ip_src', 'tp_src', 'ip_dst', 
                'tp_dst', 'ip_proto', 'icmp_code', 'icmp_type', 'flow_duration_sec', 'flow_duration_nsec',
                'idle_timeout', 'hard_timeout', 'flags', 'packet_count', 'byte_count',
                'packet_count_per_second', 'packet_count_per_nsecond', 'byte_count_per_second', 'byte_count_per_nsecond'
            ]
            
            # Check if all required columns are in the DataFrame
            for col in required_columns:
                if col not in predict_flow_dataset.columns:
                    self.logger.error(f"Missing column: {col} in prediction data")
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
            
            # Identify victim if DDoS attack detected
            victim_hosts = {}
            if ddos_traffic > 0:
                for i, pred in enumerate(y_flow_pred):
                    if pred == 1:  # If this flow is classified as DDoS
                        # Extract victim IP from the DataFrame
                        victim_ip = predict_flow_dataset.iloc[i]['ip_dst']
                        # Get the last octet which usually identifies the host in a subnet
                        try:
                            victim_id = int(str(victim_ip)[-2:]) % 20  # Last 2 digits mod 20
                            if victim_id == 0:
                                victim_id = 20  # Map 0 to 20 for host naming
                                
                            # Count occurrences of each victim
                            if victim_id in victim_hosts:
                                victim_hosts[victim_id] += 1
                            else:
                                victim_hosts[victim_id] = 1
                        except ValueError:
                            # If we can't parse the IP properly, just continue
                            continue
            
            # Log results
            self.logger.info("------------------------------------------------------------------------------")
            legitimate_percentage = (legitimate_traffic / total_predictions) * 100 if total_predictions > 0 else 0
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
                else:
                    self.logger.info("Could not determine the specific victim")
            
            self.logger.info("------------------------------------------------------------------------------")
            
            # Write to file for debugging if needed
            with open("detection_results.txt", "a") as f:
                f.write(f"{datetime.now()} - Legitimate: {legitimate_percentage:.2f}%, DDoS: {100-legitimate_percentage:.2f}%\n")
            
            # Clear buffer after prediction
            self.flow_stats_buffer = []
            
        except Exception as e:
            # Log the exception but keep it at info level
            self.logger.info(f"Error during prediction: {str(e)}")
            # For debugging, write stack trace to file but not console
            import traceback
            with open("error_log.txt", "a") as f:
                f.write(f"{datetime.now()} - {str(e)}\n")
                f.write(traceback.format_exc())
            # Don't clear buffer so we can debug if needed
