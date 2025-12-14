defmodule MachineLearningTest do
  use ExUnit.Case
  doctest MachineLearning

  test "greets the world" do
    assert MachineLearning.hello() == :world
  end
end
