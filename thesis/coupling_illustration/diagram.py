# diagram.py
from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom

cols = ("#E5F5FD", "#EBF3E7", "#ECE8F6", "#FDF7E3")

EDGE_FONT = "25"
NODE_FONT = "25"


with Diagram("", show=False, direction = "LR", graph_attr = {"size":"10,10!"}, node_attr = {"fontsize" : NODE_FONT}, outformat = "pdf", filename = "coupling_illustration"):
    
    graphene = Custom("Material", "geometry.png")
    tls = Custom("Material", "geometry.png")
    combined = Custom("Material", "geometry.png")
    combined_hamiltonian = Custom("Material", "geometry.png")


    graphene >> Edge(label="add", fontsize = EDGE_FONT)  >> combined
    tls >> Edge(label="add", fontsize = EDGE_FONT)  >> combined
    
    combined >> Edge(label="set_hamiltonian_element", fontsize = EDGE_FONT) >> combined_hamiltonian
