#!/bin/bash
./download_externals.sh
GYP=externals/downloaded/gyp/gyp
NINJA=externals/downloaded/ninja/ninja

GYP_GENERATORS=ninja ${GYP} neural_net.gyp -I neural_net.gypi \
    --toplevel-dir=`pwd` --depth=.. --no-circular-check
${NINJA} -C out/Default/ all
