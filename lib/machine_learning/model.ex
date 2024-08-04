defmodule MachineLearning.Model do
  alias MachineLearning.Layer
  import Nx.Defn

  defstruct layers: nil, execute_fun: nil, loss_fun: nil, grad_fun: nil, step: nil

  @type t :: %__MODULE__{
          layers: list(MachineLearning.Layer.t()),
          step: float(),
          execute_fun: function(),
          loss_fun: function(),
          grad_fun: function()
        }
  @type execution_with_activation :: {Nx.tensor(), list({Nx.tensor(), Nx.tensor()})}

  @doc """
  Initialize the model with the given layer sizes.

  iex> MachineLearning.Model.init([784, 16, 16, 10])
  """
  def init(layer_sizes, step \\ 0.01) do
    key = Nx.Random.key(42)
    nb_layers = Enum.count(layer_sizes) - 1

    Enum.zip(layer_sizes, Enum.drop(layer_sizes, 1))
    |> Enum.reduce({key, []}, fn {input_size, output_size}, {key, acc} ->
      {weights, key} = Nx.Random.normal(key, 0.0, 0.1, shape: {input_size, output_size})
      {biases, key} = Nx.Random.normal(key, 0.0, 0.1, shape: {output_size})
      {key, [%Layer{weights: weights, biases: biases} | acc]}
    end)
    |> elem(1)
    |> Enum.reverse()
    |> then(&%__MODULE__{layers: &1, step: step})
    |> then(&Map.put(&1, :execute_fun, EXLA.jit(execute_fun(nb_layers))))
    |> then(&Map.put(&1, :loss_fun, EXLA.jit(loss_fun(execute_fun(nb_layers)))))
    |> then(&Map.put(&1, :grad_fun, EXLA.jit(grad_fun(loss_fun(execute_fun(nb_layers))))))
  end

  @doc """
  Execute the model on the given input.
  iex> MachineLearning.Model.execute(model, input)
  """
  def execute(%__MODULE__{layers: layers}, input) do
    Enum.reduce(layers, input, fn layer, activation ->
      MachineLearning.Layer.execute(layer, activation)
    end)
  end

  def predict(model, input) do
    model
    |> to_tuple()
    |> model.execute_fun.(input)
  end

  def loss(model, input, expected) do
    model
    |> to_tuple()
    |> model.loss_fun.(input, expected)
  end

  def accuracy(model, activation, expected) do
    model
    |> predict(activation)
    |> Nx.argmax(axis: -1)
    |> Nx.equal(Nx.argmax(expected, axis: -1))
    |> Nx.mean()
  end

  def gradient(model, activations, expected) do
    model
    |> to_tuple()
    |> model.grad_fun.(activations, expected)
    |> to_list()
  end

  @spec train(t(), activations :: Nx.tensor(), expected :: Nx.tensor()) :: t()
  def train(model, activations, expected) do
    gradient = gradient(model, activations, expected)

    %{
      model
      | layers:
          Enum.zip(model.layers, gradient)
          |> Enum.map(fn {layer, gradient} -> Layer.update(layer, gradient, model.step) end)
    }
  end

  defp to_tuple(model) do
    model.layers |> List.to_tuple()
  end

  defp to_list(layers) do
    layers
    |> Tuple.to_list()
  end

  defp grad_fun(loss_fun) do
    fn model, input, expected ->
      model
      |> grad(&loss_fun.(&1, input, expected))
    end
  end

  defp loss_fun(execute_fun) do
    fn model, input, expected ->
      execute_fun.(model, input)
      |> Nx.subtract(expected)
      |> Nx.pow(2)
      |> Nx.mean(axes: [-1])
      |> Nx.sum()
    end
  end

  defp execute_fun(nb_layers) do
    1..nb_layers
    |> Enum.reduce(
      quote do
        input
      end,
      fn index, acc ->
        quote do
          unquote(acc)
          |> Nx.dot(unquote(Macro.var(:"l#{index}", nil)).weights)
          |> Nx.add(unquote(Macro.var(:"l#{index}", nil)).biases)
          |> Nx.sigmoid()
        end
      end
    )
    |> then(fn body ->
      layers_vars =
        1..nb_layers
        |> Enum.map(fn index -> Macro.var(:"l#{index}", nil) end)
        |> then(fn vars -> {:{}, [], vars} end)

      quote do
        fn unquote(layers_vars), input -> unquote(body) end
      end
    end)
    # Debug by showing function body
    # |> then(fn quoted ->
    #  IO.puts(Macro.to_string(quoted))
    #  quoted
    # end)
    |> Code.eval_quoted()
    |> elem(0)
  end
end
