{
  'targets': [
    {
      'target_name': 'neuron_tests',
      'type': 'executable',
      'dependencies': [
        '../externals/externals.gyp:gtest',
        '../neural_net.gyp:libneuralnet',
      ],
      'sources': [
        'neuron_tests.cc',
      ],
    },
    {
      'target_name': 'mfnet_tests',
      'type': 'executable',
      'dependencies': [
        '../externals/externals.gyp:gtest',
        '../neural_net.gyp:libneuralnet',
      ],
      'sources': [
        'mfnet_tests.cc',
      ],
    },
    {
      'target_name': 'genetic_alg_test',
      'type': 'executable',
      'dependencies': [
        '../externals/externals.gyp:gtest',
        '../neural_net.gyp:libneuralnet',
      ],
      'sources': [
        'genetic_alg_test.cc',
      ],
    },
    {
      'target_name': 'learner_test',
      'type': 'executable',
      'dependencies': [
        '../externals/externals.gyp:gtest',
        '../neural_net.gyp:libneuralnet',
      ],
      'sources': [
        'learner_tests.cc',
      ],
    },
  ],
}
