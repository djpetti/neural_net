{
  'target_defaults': {
    'cflags': [
      '-g',
    ],
  },
  'targets': [
    {
      'target_name': 'neuron_tests',
      'type': 'executable',
      'dependencies': [
        '<(externals):gtest',
        '<(DEPTH)/libneuralnet.gyp:*',
      ],
      'sources': [
        'neuron_tests.cc',
      ],
    },
    {
      'target_name': 'mfnet_tests',
      'type': 'executable',
      'dependencies': [
        '<(externals):gtest',
        '<(DEPTH)/libneuralnet.gyp:*',
      ],
      'sources': [
        'mfnet_tests.cc',
      ],
    },
    {
      'target_name': 'genetic_alg_test',
      'type': 'executable',
      'dependencies': [
        '<(externals):gtest',
        '<(DEPTH)/libneuralnet.gyp:*',
      ],
      'sources': [
        'genetic_alg_test.cc',
      ],
    },
    {
      'target_name': 'learner_test',
      'type': 'executable',
      'dependencies': [
        '<(externals):gtest',
        '<(DEPTH)/libneuralnet.gyp:*',
      ],
      'sources': [
        'learner_tests.cc',
      ],
    },
  ],
}
