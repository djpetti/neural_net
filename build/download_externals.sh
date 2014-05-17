#!/bin/bash

cd externals
mkdir -p downloaded
cd downloaded

# Gyp
if [ ! -d "gyp" ]
then
  svn checkout http://gyp.googlecode.com/svn/trunk/ gyp
fi

# Ninja
if [ ! -d "ninja" ]
then
  git clone https://github.com/martine/ninja.git
fi
if [ ! -f "ninja/ninja" ]
then
  cd ninja
  ./bootstrap.py
  cd ..
fi

# Gtest
GTEST_VERSION=1.7.0
GTEST_NAME=gtest-${GTEST_VERSION}
if [ ! -f "${GTEST_NAME}.zip" ]
then
  wget -O ${GTEST_NAME}.zip \
      https://googletest.googlecode.com/files/${GTEST_NAME}.zip
fi
if [ ! -d "${GTEST_NAME}" ]
then
  unzip ${GTEST_NAME}.zip
fi
