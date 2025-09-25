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


with Diagram("", show=False, direction = "LR", graph_attr = {"size":"10,10!"}, node_attr = {"fontsize" : NODE_FONT}, outformat = "pdf", filename = "granad_schema"):
    with Cluster("Building Blocks", graph_attr = {"bgcolor" : cols[0]} ) as cluster1:
        cluster1.dot.graph_attr["bgcolor"] = cols[0]
        cluster1.dot.graph_attr["fontsize"] = NODE_FONT

        material, orbital = Custom("Material", "geometry.png"), Custom("Orbital", "pz_orbital_v2.png")

    with Cluster("Structure Specification") as cluster2:
        cluster2.dot.graph_attr["bgcolor"] = cols[1]
        cluster2.dot.graph_attr["fontsize"] = NODE_FONT

        orb1 = Custom("OrbitalList", "single.png")
        orb2 = Custom("OrbitalList", "double.png")
        orbs = [orb1 >> Edge(label="Rotate, Dope, Combine", fontsize = EDGE_FONT) >> orb2]
        material >> Edge(label="cut", fontsize = EDGE_FONT) >> orb1
        orbital >> Edge(label="append", fontsize = EDGE_FONT) >> orb1

    with Cluster("Simulation Output") as cluster3:
        cluster3.dot.graph_attr["bgcolor"] = cols[2]
        cluster3.dot.graph_attr["fontsize"] = NODE_FONT

        output_time, output_freq, output_static = Custom("γ(t)\n\n", "", fontsize = "30"), Custom("G(ω)\n\n", "", fontsize = "30"), Custom("ε\n\n", "", fontsize = "30")
        orbs >> Edge(label="Dynamical Simulation", fontsize = EDGE_FONT) >> output_time
        orbs >> Edge(label="Green's Function", fontsize = EDGE_FONT) >> output_freq
        orbs >> Edge(label="Independent Particle", fontsize = EDGE_FONT) >> output_static    
