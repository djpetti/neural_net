#!/bin/bash

cd externals
mkdir -p downloaded
cd downloaded

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
