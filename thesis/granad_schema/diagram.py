# diagram.py
from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB

from diagrams import Diagram, Cluster
from diagrams.custom import Custom    

with Diagram("Web Service", show=False):
    ELB("lb") >> EC2("web") >> RDS("userdb")

with Diagram("Custom with local icons\n Can be downloaded here: \nhttps://creativecommons.org/about/downloads/", show=False, filename="custom_local", direction="LR"):
  cc_heart = Custom("Creative Commons", "./my_resources/cc_heart.black.png")
  cc_attribution = Custom("Credit must be given to the creator", "./my_resources/cc_attribution.png")

  cc_sa = Custom("Adaptations must be shared\n under the same terms", "./my_resources/cc_sa.png")
  cc_nd = Custom("No derivatives or adaptations\n of the work are permitted", "./my_resources/cc_nd.png")
  cc_zero = Custom("Public Domain Dedication", "./my_resources/cc_zero.png")

  with Cluster("Non Commercial"):
    non_commercial = [Custom("Y", "./my_resources/cc_nc-jp.png") - Custom("E", "./my_resources/cc_nc-eu.png") - Custom("S", "./my_resources/cc_nc.png")]

  cc_heart >> cc_attribution
  cc_heart >> non_commercial
  cc_heart >> cc_sa
  cc_heart >> cc_nd
  cc_heart >> cc_zero    
