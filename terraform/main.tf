# SERVICE 1: ONNX RUNTIME INFERENCE
# Condition: Active only when benchmark_mode is set to 'onnx'
resource "aws_ecs_service" "onnx" {
  name            = "inventory-monitor-onnx-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.onnx.arn
  launch_type     = "FARGATE"

  # 1 instance if mode is 'onnx', otherwise 0 to save costs
  desired_count = var.benchmark_mode == "onnx" ? 1 : 0

  network_configuration {
    subnets          = [aws_subnet.public.id]
    security_groups  = [aws_security_group.api_sg.id]
    assign_public_ip = true
  }
}

# SERVICE 2: PYTORCH NATIVE INFERENCE
# Condition: Active only when benchmark_mode is set to 'pytorch'
resource "aws_ecs_service" "pytorch" {
  name            = "inventory-monitor-pytorch-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.pytorch.arn
  launch_type     = "FARGATE"

  # 1 instance if mode is 'pytorch', otherwise 0
  desired_count = var.benchmark_mode == "pytorch" ? 1 : 0

  network_configuration {
    subnets          = [aws_subnet.public.id]
    security_groups  = [aws_security_group.api_sg.id]
    assign_public_ip = true
  }
}
