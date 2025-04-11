# Terraform configuration for AWS Infrastructure as Code (IaC)
# This defines an AWS infrastructure setup for deploying ML models using Docker and Kubernetes.

provider "aws" {
  region = "us-east-1"
}

# S3 Bucket for Model Storage
resource "aws_s3_bucket" "model_storage" {
  bucket = "ml-model-storage-mtejada"
  acl    = "private"
}

# Elastic Container Registry (ECR) for storing Docker images
resource "aws_ecr_repository" "ml_repo" {
  name = "ml-model-repo"
}

# IAM Role for EC2 to access S3 and ECR
resource "aws_iam_role" "ec2_role" {
  name = "ml_ec2_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = { Service = "ec2.amazonaws.com" },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy_attachment" "ec2_s3_access" {
  name       = "ec2_s3_access"
  roles      = [aws_iam_role.ec2_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_policy_attachment" "ec2_ecr_access" {
  name       = "ec2_ecr_access"
  roles      = [aws_iam_role.ec2_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
}

# EC2 Instance for Running Model Inference
resource "aws_instance" "ml_server" {
  ami           = "ami-0c55b159cbfafe1f0" # Ubuntu AMI, replace with latest
  instance_type = "t2.micro"
  key_name      = "ml-key-pair" # Ensure you have this key pair created

  iam_instance_profile = aws_iam_role.ec2_role.name

  tags = {
    Name = "ML-Inference-Server"
  }
}

# Output values for reference
output "s3_bucket_name" {
  value = aws_s3_bucket.model_storage.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.ml_repo.repository_url
}

output "ec2_public_ip" {
  value = aws_instance.ml_server.public_ip
}
