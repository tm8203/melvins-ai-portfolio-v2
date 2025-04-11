# Terraform Documentation

This document explains the configuration and components used within the Terraform setup to deploy and manage the infrastructure for the AI portfolio application.

## Files Overview

- **`provider.tf`**: Configures the AWS provider to connect to the AWS cloud environment, specifying necessary credentials and the region where the resources will be created.
  
- **`variables.tf`**: Defines the input variables used in the Terraform configuration, such as region, instance type, etc.
  
- **`outputs.tf`**: Specifies the outputs Terraform will provide once the infrastructure is deployed. This is useful to display important information like public IP addresses and other resource details.

## `provider.tf`

```hcl
provider "aws" {
  region = var.aws_region
}
