defmodule MachineLearning.CorpusTest do
  use ExUnit.Case
  doctest MachineLearning.Corpus

  alias MachineLearning.Corpus

  @corpus_path "./tmp/corpus"

  describe "load_texts/2" do
    test "loads texts from corpus directory if it exists" do
      if File.dir?(@corpus_path) do
        texts = Corpus.load_texts(@corpus_path, max_files: 5)

        assert is_list(texts)
        assert length(texts) > 0
        assert Enum.all?(texts, &is_binary/1)
      else
        # Skip test if corpus doesn't exist
        assert true
      end
    end

    test "returns empty list for non-existent directory" do
      texts = Corpus.load_texts("./non_existent_dir")
      assert texts == []
    end

    test "filters by minimum length" do
      if File.dir?(@corpus_path) do
        texts = Corpus.load_texts(@corpus_path, max_files: 10, min_length: 100)

        assert Enum.all?(texts, fn text -> String.length(text) >= 100 end)
      else
        assert true
      end
    end
  end

  describe "load_split_texts/2" do
    test "splits texts by lines" do
      if File.dir?(@corpus_path) do
        lines =
          Corpus.load_split_texts(
            @corpus_path,
            max_files: 5,
            split_by: :lines,
            min_length: 10
          )

        assert is_list(lines)
        assert length(lines) > 0

        # Verify no empty lines
        assert Enum.all?(lines, fn line ->
                 String.trim(line) != "" && String.length(line) >= 10
               end)
      else
        assert true
      end
    end

    test "splits texts by paragraphs" do
      if File.dir?(@corpus_path) do
        paragraphs =
          Corpus.load_split_texts(
            @corpus_path,
            max_files: 5,
            split_by: :paragraphs,
            min_length: 20
          )

        assert is_list(paragraphs)
        # Paragraphs should generally be longer than lines
        assert Enum.all?(paragraphs, fn p -> String.length(p) >= 20 end)
      else
        assert true
      end
    end
  end

  describe "load_chunked_texts/2" do
    test "chunks texts into fixed sizes" do
      if File.dir?(@corpus_path) do
        chunks =
          Corpus.load_chunked_texts(
            @corpus_path,
            max_files: 3,
            chunk_size: 200,
            overlap: 20
          )

        assert is_list(chunks)
        assert length(chunks) > 0

        # Most chunks should be around the target size
        # (last chunk of each file might be shorter)
        avg_length = chunks |> Enum.map(&String.length/1) |> Enum.sum() |> div(length(chunks))
        assert avg_length > 100
        assert avg_length < 250
      else
        assert true
      end
    end
  end

  describe "load_sample/2" do
    test "loads a sample of texts" do
      if File.dir?(@corpus_path) do
        sample = Corpus.load_sample(@corpus_path, 20)

        assert is_list(sample)
        assert length(sample) <= 20
        assert Enum.all?(sample, &is_binary/1)
      else
        assert true
      end
    end
  end

  describe "stats/1" do
    test "prints statistics without errors" do
      if File.dir?(@corpus_path) do
        # Just verify it doesn't crash
        assert :ok = Corpus.stats(@corpus_path)
      else
        assert true
      end
    end
  end
end
