-- lstm cell with peepholes
function lstmp(input_size, x, prev_c, prev_h, rnn_size)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(input_size, 4*rnn_size)(x)
  local h2h = nn.LinearNoBias(rnn_size, 4*rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)

  local in_gate          = nn.Sigmoid()(
    nn.CAddTable()({
      nn.SelectTable(1)(sliced_gates),
      nn.CMul(rnn_size)(prev_c)
    })
  )

  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(
    nn.CAddTable()({
      nn.SelectTable(3)(sliced_gates),
      nn.CMul(rnn_size)(prev_c)
    })
  )

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })

  local out_gate        = nn.Sigmoid()(
    nn.CAddTable()({
      nn.SelectTable(4)(sliced_gates),
      nn.CMul(rnn_size)(next_c)
    })
  )

  local next_h          = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end
