from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB, Route53
from diagrams.aws.devtools import Codepipeline
from diagrams.onprem.container import Docker
from diagrams.onprem.iac import Terraform
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.network import Nginx

# Define the architecture diagram
with Diagram("AI Model Deployment Architecture", show=False, filename="Portfolio_Home_Files/architecture_diagram"):
    
    # Load Balancer & Domain
    dns = Route53("DNS")
    lb = ELB("Load Balancer")

    # Define Terraform & CI/CD pipeline
    with Cluster("Infrastructure as Code"):
        terraform = Terraform("Terraform")
        github = GithubActions("GitHub Actions")
        ci_cd = CodePipeline("CI/CD Pipeline")
        terraform >> github >> ci_cd

    # Define Application Cluster
    with Cluster("AI Model Deployment"):
        web_server = Nginx("Nginx Reverse Proxy")
        app = EC2("Streamlit App")
        model = Docker("AI Model (Docker)")
        web_server >> app >> model

    # Database
    db = RDS("Database")

    # Architecture Connections
    dns >> lb >> web_server
    model >> db
    ci_cd >> app
    terraform >> [app, db, lb]
