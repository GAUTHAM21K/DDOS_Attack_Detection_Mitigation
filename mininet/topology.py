from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import OVSKernelSwitch, RemoteController
import time
import os
import sys

class MyTopo(Topo):
    def build(self):
        # Create 6 switches with OpenFlow 13
        s1 = self.addSwitch('s1', cls=OVSKernelSwitch, protocols='OpenFlow13')
        s2 = self.addSwitch('s2', cls=OVSKernelSwitch, protocols='OpenFlow13')
        s3 = self.addSwitch('s3', cls=OVSKernelSwitch, protocols='OpenFlow13')
        s4 = self.addSwitch('s4', cls=OVSKernelSwitch, protocols='OpenFlow13')
        s5 = self.addSwitch('s5', cls=OVSKernelSwitch, protocols='OpenFlow13')
        s6 = self.addSwitch('s6', cls=OVSKernelSwitch, protocols='OpenFlow13')
        
        # Create hosts (3 hosts per switch)
        for i in range(1, 19):
            switch_num = ((i - 1) // 3) + 1
            host = self.addHost(f'h{i}', 
                              cpu=1.0/20,
                              mac=f"00:00:00:00:00:{i:02d}", 
                              ip=f"10.0.0.{i}/24")
            
            # Connect host to its switch
            self.addLink(host, locals()[f's{switch_num}'])
        
        # Connect switches in a line: s1 -- s2 -- s3 -- s4 -- s5 -- s6
        self.addLink(s1, s2)
        self.addLink(s2, s3)
        self.addLink(s3, s4)
        self.addLink(s4, s5)
        self.addLink(s5, s6)

def startNetwork():
    """
    Create and start the network with controller connection
    """
    # Set up the topology
    topo = MyTopo()
    
    # Use the specified controller IP and port
    controller_ip = '127.0.0.1'  # Default to localhost
    if len(sys.argv) >= 2:
        controller_ip = sys.argv[1]
        
    # Create the controller
    c0 = RemoteController('c0', ip=controller_ip, port=6653)
    
    # Create the Mininet with the topology and controller
    net = Mininet(topo=topo, link=TCLink, controller=c0)
    
    # Start the network
    print("*** Starting network")
    net.start()
    
    # Wait for network to initialize
    print("*** Waiting for SDN controller to connect...")
    time.sleep(5)
    
    # Create directory for script outputs
    if not os.path.exists("traffic_scripts"):
        os.makedirs("traffic_scripts")
    
    # Create normal traffic generator script
    with open("traffic_scripts/normal_traffic.sh", "w") as f:
        f.write("""#!/bin/bash
# Generate normal traffic between hosts
while true; do
    # Ping with normal intervals
    ping -c 5 10.0.0.10 &
    ping -c 5 10.0.0.15 &
    # HTTP-like traffic
    for i in {1..5}; do
        echo "GET /index.html HTTP/1.1\\nHost: 10.0.0.$i\\n\\n" | nc -w 1 10.0.0.$i 80 &
    done
    sleep 10
done
""")
    
    # Create DDoS attack script
    with open("traffic_scripts/ddos_attack.sh", "w") as f:
        f.write("""#!/bin/bash
# Target IP - victim
TARGET_IP="10.0.0.10"
echo "Starting DDoS attack simulation against $TARGET_IP"
# Launch various attack types
# SYN flood
hping3 --flood --rand-source -S -p 80 $TARGET_IP &
# UDP flood to DNS port
hping3 --flood --udp -p 53 $TARGET_IP &
# ICMP flood
ping -f $TARGET_IP &
echo "DDoS attack simulation is running. Press Ctrl+C to stop."
""")
    
    # Make scripts executable
    os.system("chmod +x traffic_scripts/normal_traffic.sh")
    os.system("chmod +x traffic_scripts/ddos_attack.sh")
    
    print("\n*** Network setup complete!")
    print("*** To generate normal traffic run: h1 ./traffic_scripts/normal_traffic.sh &")
    print("*** To simulate DDoS attack run: h7 ./traffic_scripts/ddos_attack.sh &")
    print("*** DDoS detection should trigger automatic mitigation\n")
    
    # Start CLI
    CLI(net)
    
    # After CLI is closed, stop the network
    print("*** Stopping network")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    startNetwork()
