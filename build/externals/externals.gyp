{
  'targets': [
  {
    'target_name': 'gtest',
    'type': 'static_library',
    'sources': [
      'downloaded/gtest-1.7.0/fused-src/gtest/gtest-all.cc',
      'downloaded/gtest-1.7.0/fused-src/gtest/gtest_main.cc',
    ],
    'include_dirs': [
      'downloaded/gtest-1.7.0/include',
    ],
    'cflags': [
      '-Werror',
      '-Wall',
    ],
    'direct_dependent_settings': {
      'include_dirs': [
        'downloaded/gtest-1.7.0/include',
      ],
      'ldflags': [
        '-pthread',
      ],
    },
  },
  ],
}
