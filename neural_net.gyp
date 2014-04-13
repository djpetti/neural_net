{
  'targets': [
    {
      'target_name': 'neural_net_all',
      'type': 'none',
      'dependencies': [
        'tests/tests.gyp:*',
      ],
    },
    {
      'target_name': 'libneuralnet',
      'type': 'static_library',
      'cflags': [
        '-std=c++11',
        '-Werror',
        '-Wall',
      ],
      'sources': [
        'genetic_algorithm.cc',
        'multilayered_feedforward.cc',
        'neuron.cc',
        'output_functions.cc',
      ],
    },
  ],
}
