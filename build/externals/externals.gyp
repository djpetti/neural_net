{
  'targets': [
  {
    'target_name': 'gtest',
    'type': 'static_library',
    'sources': [
      'gtest-1.7.0/fused-src/gtest/gtest-all.cc',
      'gtest-1.7.0/fused-src/gtest/gtest_main.cc',
    ],
    'include_dirs': [
      'gtest-1.7.0/include',
    ],
    'cflags': [
      '-Werror',
      '-Wall',
    ],
    'direct_dependent_settings': {
      'include_dirs': [
        'gtest-1.7.0/include',
      ],
      'ldflags': [
        '-pthread',
      ],
      'cflags': [
        # For testing, we generally want this.
        '-std=c++11',
        '-g',
        '-O0',
      ],
    },
  },
  ],
}
