tokens = File.read!("vocabulary.bert") |> :erlang.binary_to_term() |> Enum.map(&MachineLearning.BytePairEncoding.Token.new/1)

save_tokens = fn tokens ->
  File.write!("vocabulary.bert", :erlang.term_to_binary(Enum.map(tokens, & &1.value)))
end
