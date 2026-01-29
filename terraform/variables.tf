variable "benchmark_mode" {
  description = "Determines which model to run. Options: 'onnx', 'pytorch', 'none'"
  type        = string
  default     = "onnx"
}
