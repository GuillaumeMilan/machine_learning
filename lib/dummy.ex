defmodule Dummy do
  def test do
    m1 =
      Axon.input("input_ids", shape: {nil, nil})
      |> Axon.dense(1)

    {init_fn, pred_fn} = Axon.build(m1, compiler: EXLA)
    seq_len = 3
    input_template = Nx.template({1, seq_len}, :s64)

    params = init_fn.(%{"input_ids" => input_template}, Axon.ModelState.new(%{}))

    pred_fn.(params, %{"input_ids" => Nx.tensor([[1, 2, 3]])})
  end
end
