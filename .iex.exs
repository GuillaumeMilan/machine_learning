tokens = File.read!("vocabulary.bert") |> :erlang.binary_to_term() |> Enum.map(&MachineLearning.BytePairEncoding.Token.new/1)
