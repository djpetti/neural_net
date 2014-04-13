#!/bin/bash
GYP_GENERATORS=ninja gyp neural_net.gyp --toplevel-dir=`pwd` --depth=0 \
    --no-circular-check
ninja -C out/Default/ all
