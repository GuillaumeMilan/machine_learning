import Config

config :nx, :default_backend, EXLA.Backend

# EXLA configuration for macOS
# Note: GPU support on macOS requires EXLA to be compiled with Metal support
# For EXLA 0.10.0, the available platforms are :host (CPU) and :cuda (NVIDIA GPU)
# Metal support is experimental and requires building from source with XLA_TARGET=metal_arm64
config :exla, :clients,
  host: [platform: :host]

config :exla, :default_client, :host
