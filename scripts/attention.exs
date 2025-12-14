set =
  MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte", 10)

aw1 = MachineLearning.Network.init([784, 64, 8], 0.01)
aw2 = MachineLearning.Network.init([784, 64, 8], 0.01)
aw3 = MachineLearning.Network.init([784, 64, 8], 0.01)
aw4 = MachineLearning.Network.init([784, 64, 8], 0.01)
cast_layer = MachineLearning.Network.init([8 * 4, 10], 0.01)
layers = {aw1, aw2, aw3, aw4, cast_layer}

input_var = Macro.var(:input, nil)
concatenated_var = Macro.var(:concatenated, nil)
aw1_var = Macro.var(:aw1, nil)
aw2_var = Macro.var(:aw2, nil)
aw3_var = Macro.var(:aw3, nil)
aw4_var = Macro.var(:aw4, nil)
aw5_var = Macro.var(:aw5, nil)
aw6_var = Macro.var(:aw6, nil)
aw7_var = Macro.var(:aw7, nil)
aw8_var = Macro.var(:aw8, nil)
cast_layer_var = Macro.var(:cast_layer, nil)

execute_fun_non_jit =
  quote do
    fn {unquote(aw1_var), unquote(aw2_var), unquote(aw3_var), unquote(aw4_var),
        unquote(cast_layer_var)},
       unquote(input_var) ->
      out1 = unquote(MachineLearning.Network.quote_execution(input_var, Macro.var(:aw1, nil), 2))
      out2 = unquote(MachineLearning.Network.quote_execution(input_var, Macro.var(:aw2, nil), 2))
      out3 = unquote(MachineLearning.Network.quote_execution(input_var, Macro.var(:aw3, nil), 2))
      out4 = unquote(MachineLearning.Network.quote_execution(input_var, Macro.var(:aw4, nil), 2))

      unquote(concatenated_var) =
        Nx.concatenate([out1, out2, out3, out4], axis: 1)

      unquote(
        MachineLearning.Network.quote_execution(
          concatenated_var,
          Macro.var(:cast_layer, nil),
          1
        )
      )
    end
  end
  # |> tap(fn quoted ->
  #   IO.puts("Generated execution function:")
  #   IO.puts(Macro.to_string(quoted))
  # end)
  |> Code.eval_quoted()
  |> elem(0)

loss_fun_non_jit = fn network, input, expected ->
  execute_fun_non_jit.(network, input)
  |> Nx.subtract(expected)
  |> Nx.pow(2)
  |> Nx.mean(axes: [-1])
  |> Nx.sum()
end

import Nx.Defn, only: [grad: 2]

grad_fun_non_jit = fn layers, input, expected ->
  layers
  |> grad(&loss_fun_non_jit.(&1, input, expected))
end

accuracy_fun_non_jit = fn network, input, expected ->
  execute_fun_non_jit.(network, input)
  |> Nx.argmax(axis: -1)
  |> Nx.equal(Nx.argmax(expected, axis: -1))
  |> Nx.mean()
end

execute_fun = execute_fun_non_jit |> EXLA.jit()
loss_fun = loss_fun_non_jit |> EXLA.jit()
grad_fun = grad_fun_non_jit |> EXLA.jit()
accuracy_fun = accuracy_fun_non_jit |> EXLA.jit()

train_fun = fn layers, inputs, expected ->
  gradients =
    grad_fun.(layers, inputs, expected)
    |> Tuple.to_list()

  layers
  |> Tuple.to_list()
  |> Enum.zip(gradients)
  |> Enum.map(fn {layer, grad} ->
    MachineLearning.Network.apply_gradient(layer, grad)
  end)
  |> List.to_tuple()
end

model =
  set
  |> Enum.with_index()
  |> Enum.reduce(layers, fn {{image, expected}, index}, model ->
    if rem(index, 100) == 0 do
      IO.puts("Index #{index}")

      loss_fun.(model, image, expected)
      |> Nx.to_number()
      |> IO.inspect(label: "Loss")

      accuracy_fun.(model, image, expected)
      |> Nx.to_number()
      |> IO.inspect(label: "Accuracy")

      # grad_fun.(model, image, expected)
      # |> IO.inspect(label: "Gradient")
    end

    train_fun.(model, image, expected)
  end)
