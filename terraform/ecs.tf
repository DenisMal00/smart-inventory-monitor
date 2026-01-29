# Logical cluster to group our inference services
resource "aws_ecs_cluster" "main" {
  name = "inventory-monitor-cluster"

  # Enable CloudWatch Container Insights for detailed RAM/CPU metrics (Crucial for CV benchmarks)
  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Project = "smart-inventory"
  }
}
