data "aws_caller_identity" "current" {}

resource "aws_cloudwatch_log_group" "inventory_logs" {
  name              = "/ecs/inventory-monitor"
  retention_in_days = 7
}

# TASK 1: ONNX RUNTIME (Configured for Benchmark Parity)
# Target: Same resources as PyTorch to compare raw inference speed per CPU cycle.
resource "aws_ecs_task_definition" "onnx" {
  family                   = "inventory-monitor-onnx"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512 # 1 vCPU
  memory                   = 1024 # 2 GB

  execution_role_arn = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "inference-api"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.eu-central-1.amazonaws.com/inventory-monitor:onnx-int8"
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.inventory_logs.name
          "awslogs-region"        = "eu-central-1"
          "awslogs-stream-prefix" = "onnx"
        }
      }
    }
  ])
}

# TASK 2: PYTORCH NATIVE (Baseline)
# Target: Performance comparison.
resource "aws_ecs_task_definition" "pytorch" {
  family                   = "inventory-monitor-pytorch"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024 # 1 vCPU
  memory                   = 2048 # 2 GB

  execution_role_arn = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "inference-api"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.eu-central-1.amazonaws.com/inventory-monitor:pytorch"
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.inventory_logs.name
          "awslogs-region"        = "eu-central-1"
          "awslogs-stream-prefix" = "pytorch"
        }
      }
    }
  ])
}
