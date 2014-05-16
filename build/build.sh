#!/bin/bash
./download_externals.sh
GYP_GENERATORS=ninja gyp neural_net.gyp -I neural_net.gypi \
    --toplevel-dir=`pwd` --depth=.. --no-circular-check
ninja -C out/Default/ all
