#!/usr/bin/env elixir

# Quick demonstration of loading training texts from corpus
#
# Usage:
#   elixir scripts/corpus_demo.exs

IO.puts("=== Corpus Loading Demonstration ===\n")

corpus_dir = "./tmp/corpus"

if File.dir?(corpus_dir) do
  IO.puts("Corpus directory found: #{corpus_dir}\n")

  # Show statistics
  IO.puts("1. Corpus Statistics:")
  MachineLearning.Corpus.stats(corpus_dir)

  # Load sample
  IO.puts("\n2. Loading sample texts...")
  sample = MachineLearning.Corpus.load_sample(corpus_dir, 5)

  IO.puts("Loaded #{length(sample)} sample texts\n")

  sample
  |> Enum.take(3)
  |> Enum.with_index(1)
  |> Enum.each(fn {text, idx} ->
    preview = text |> String.slice(0..100) |> String.replace("\n", " ")
    IO.puts("Sample #{idx}: #{preview}...")
  end)

  # Load by lines
  IO.puts("\n3. Loading split by lines...")

  lines =
    MachineLearning.Corpus.load_split_texts(
      corpus_dir,
      split_by: :lines,
      max_files: 10,
      min_length: 30
    )

  IO.puts("Loaded #{length(lines)} lines")
  IO.puts("First 3 lines:")

  lines
  |> Enum.take(3)
  |> Enum.each(fn line ->
    IO.puts("  - #{String.slice(line, 0..80)}#{if String.length(line) > 80, do: "...", else: ""}")
  end)

  # Load chunked
  IO.puts("\n4. Loading chunked texts...")

  chunks =
    MachineLearning.Corpus.load_chunked_texts(
      corpus_dir,
      chunk_size: 300,
      overlap: 30,
      max_files: 5
    )

  IO.puts("Created #{length(chunks)} chunks")

  chunk_lengths = Enum.map(chunks, &String.length/1)
  avg_length = Enum.sum(chunk_lengths) / length(chunk_lengths)

  IO.puts("Average chunk size: #{trunc(avg_length)} characters")
  IO.puts("Min chunk size: #{Enum.min(chunk_lengths)} characters")
  IO.puts("Max chunk size: #{Enum.max(chunk_lengths)} characters")

  IO.puts("\n=== Demonstration Complete ===")
else
  IO.puts("Corpus directory not found: #{corpus_dir}")
  IO.puts("\nTo create a corpus, run:")
  IO.puts("  MachineLearning.BytePairEncoding.add_to_corpus(\"./lib\", \"./tmp/corpus\")")
end
