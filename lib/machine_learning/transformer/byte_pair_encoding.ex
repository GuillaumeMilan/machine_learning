defmodule MachineLearning.Transformer.BytePairEncoding do
  def initialize(corpus_dir, cache_dir, opts \\ %{}) do
    inital_tokens = Map.get(opts, :tokens, [])

    list_files(corpus_dir)
    |> Stream.transform([], fn elem, acc ->
      nil
    end)
  end

  # Sort of recursive LS
  @spec list_files(Path.t()) :: list(Path.t())
  defp list_files(directory) do
    File.ls!(directory)
    |> Enum.reject(&String.starts_with?(&1, "."))
    # Reject node_modules and similar directories
    |> Enum.reject(&String.ends_with?(&1, "_modules"))
    |> Stream.flat_map(fn name ->
      full_name = Path.join(directory, name)

      case File.dir?(full_name) do
        true -> list_files(full_name)
        false -> [full_name]
      end
    end)
  end
end
