# terraform/security.tf

resource "aws_security_group" "api_sg" {
  name        = "inventory-monitor-api-sg"
  description = "Allow inbound traffic to FastAPI on port 8000"
  vpc_id      = aws_vpc.main.id

  # Inbound rule for FastAPI
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Open to the world for benchmarking
  }

  # Outbound rule (allow container to reach ECR and CloudWatch)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

