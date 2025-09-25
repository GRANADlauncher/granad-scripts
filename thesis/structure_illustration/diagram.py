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
    
    hbn = Custom("Material", "geometry.png")
    hbn_rotated = Custom("Material", "geometry.png")
    graphene = Custom("Material", "geometry.png")
    graphene_defective = Custom("Material", "geometry.png")
    orbital = Custom("Material", "geometry.png")    
    stack = Custom("Material", "geometry.png")

    hbn >> Edge(label="flake.rotate", fontsize = EDGE_FONT) >> hbn_rotated
    graphene >> Edge(label="del flake[0]", fontsize = EDGE_FONT) >> graphene_defective

    hbn_rotated >> Edge(label="stack", fontsize = EDGE_FONT) >> stack
    graphene_defective >> Edge(label="stack", fontsize = EDGE_FONT) >> stack
    orbital >> Edge(label="stack", fontsize = EDGE_FONT) >> stack
