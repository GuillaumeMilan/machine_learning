import Config

config :nx, :default_backend, EXLA.Backend

config :exla, :clients,
  mps: [platform: :mps],
  host: [platform: :host]

# Use Metal GPU by default (fallback to :host if issues occur)
config :exla, :default_client, :mps
