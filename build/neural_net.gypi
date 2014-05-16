{
  'target_defaults': {
    'cflags': [
      '-std=c++11',
      '-Werror',
      '-Wall',
    ],
  },
  'variables': {
    'externals': '<(DEPTH)/build/externals/externals.gyp',
  },
}
