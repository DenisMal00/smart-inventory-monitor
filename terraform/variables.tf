variable "benchmark_mode" {
  description = "Determines which model to run. Options: 'onnx', 'pytorch', 'none'"
  type        = string
  default     = "onnx"
}

variable "duckdns_token" {
  type      = string
  sensitive = true
}

variable "duckdns_domain" {
  type      = string
}
