training_conf = %{epoch: 2, corpus_dir: "/tmp/elixir/", sample_size: 10_000, log_level: :debug}
conf = %{max_seq_len: 128, embed_dim: 128, num_heads: 4, num_layers: 2, ff_dim: 512}
MachineLearning.TransformerTraining.run(Map.merge(conf, training_conf))
