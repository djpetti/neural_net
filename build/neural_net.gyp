{
  'targets': [
    {
      'target_name': 'neural_net_all',
      'type': 'none',
      'dependencies': [
        '<(DEPTH)/tests/tests.gyp:*',
        '<(DEPTH)/libneuralnet.gyp:*',
      ],
    },
  ],
}
