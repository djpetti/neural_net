{
  'targets': [
    {
      'target_name': 'libneuralnet',
      'type': 'static_library',
      'sources': [
        'genetic_algorithm.cc',
        'logger.cc',
        'multilayered_feedforward.cc',
        'neuron.cc',
        'output_functions.cc',
        'supervised_learner.cc',
      ],
    },
  ],
}
