defmodule MachineLearning.Cost do
  @moduledoc """
  Back propagation group helpers to train a model.
  """
  alias MachineLearning.Model
  alias MachineLearning.Layer
  @type gradient :: list(Layer.t())

  @doc """
  Calculate the cost of the model.
  """
  @spec value(Nx.tensor(), Nx.tensor()) :: Nx.tensor()
  def value(output, expected) do
    Nx.subtract(output, expected)
    |> Nx.pow(2)
    |> Nx.sum()
  end

  @doc """
  Calculate the cost gradient of the model on one input.
  """
  @spec gradient(Model.t(), Model.execution_with_activation(), Nx.tensor()) ::
          list(%{bias: Nx.tensor(), weights: Nx.tensor()})
  def gradient(%Model{layers: layers}, {last_activation, outputs}, expected) do
    # DCost / DOutput = 2 * (Output - Expected)
    initial_activation_derivative =
      Nx.subtract(last_activation, expected)
      |> Nx.multiply(2)

    # Calculate the cost gradient for each layer
    {_, gradients} =
      outputs
      |> Enum.zip(layers |> Enum.reverse())
      |> Enum.reduce(
        {initial_activation_derivative, []},
        fn {{activation, previous_activation}, layer}, {activation_derivative, acc} ->
          bias_derivative = activation_derivative |> sigmoid_derivative()

          weight_derivative =
            cost_weight_derivative(bias_derivative, activation, previous_activation)

          activation_derivative = cost_activation_derivative(bias_derivative, layer.weights)

          # Add the layer gradient to the accumulator
          {activation_derivative,
           [%Layer{biases: bias_derivative, weights: weight_derivative} | acc]}
        end
      )

    gradients
  end

  defp cost_weight_derivative(bias_derivative, activation, previous_activation) do
    1..Nx.size(previous_activation)
    |> Enum.map(fn _ ->
      bias_derivative
    end)
    |> Nx.stack()
    |> Nx.transpose()
    |> Nx.tensor()
    # DZ / DWeights = Activation of previous layer
    |> Nx.multiply(
      # Activation of previous layer extended to the shape of weights
      # TODO check if there is a more efficient way to do this with Nx.transpose and Nx.stack
      1..Nx.size(activation)
      |> Enum.map(fn _ ->
        previous_activation
        |> Nx.tensor()
      end)
      |> Nx.stack()
    )
  end

  defp cost_activation_derivative(bias_derivative, weights) do
    weights
    |> Nx.transpose()
    # DZ / DActivation = Weights
    |> Nx.dot(bias_derivative)
  end

  # Derivative expression of the sigmoid function
  defp sigmoid_derivative(t) do
    s = t |> Nx.sigmoid()
    s |> Nx.multiply(s |> Nx.multiply(-1) |> Nx.add(1))
  end
end
