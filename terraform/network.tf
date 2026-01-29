# terraform/network.tf

# Main VPC for the Smart Inventory project
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "inventory-monitor-vpc"
  }
}

# Internet Gateway to provide internet access to the VPC
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "inventory-monitor-igw"
  }
}

# Public Subnet for ECS Fargate tasks.
# NOTE: Using a public subnet with a Security Group is a deliberate choice
# to optimize costs by avoiding NAT Gateway fees (~$32/mo).
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "eu-central-1a"
  map_public_ip_on_launch = true

  tags = {
    Name = "inventory-monitor-public-subnet"
  }
}

# Route table to direct internet-bound traffic (0.0.0.0/0) to the Internet Gateway
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "inventory-monitor-public-rt"
  }
}

# Associate the public subnet with the public route table
resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}