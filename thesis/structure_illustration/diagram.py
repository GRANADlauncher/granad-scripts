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


with Diagram("", show=False, direction = "LR", graph_attr = {"size":"10,10!"}, node_attr = {"fontsize" : NODE_FONT}, outformat = "pdf", filename = "structure_illustration"):
    
    hbn = Custom("hBN", "hbn.png", height="2", width="3")
    hbn_rotated = Custom("", "hbn_rotated.png", height="2", width="3")
    graphene = Custom("Graphene", "graphene.png", height="2", width="3")
    graphene_defective = Custom("", "graphene_defective.png", height="2", width="3")
    orbital = Custom("Isolated Orbital", "orbital.png", height="2", width="3")    
    stack = Custom("", "stack.png", height="2", width="3")

    hbn >> Edge(label="flake.rotate", fontsize = EDGE_FONT) >> hbn_rotated
    graphene >> Edge(label="del flake[index]", fontsize = EDGE_FONT) >> graphene_defective

    hbn_rotated >> Edge(label="add", fontsize = EDGE_FONT) >> stack
    graphene_defective >> Edge(label="add", fontsize = EDGE_FONT) >> stack
    orbital >> Edge(label="append", fontsize = EDGE_FONT) >> stack
