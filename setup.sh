#!/bin/bash




# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command

ARCH=x64
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo $SETUP_DIR
export XCFROOTDIR=$SETUP_DIR

export ASIROOTDIR="$XCFROOTDIR/ASI_camera/daq/asi_sdk"

echo "XCFROOTDIR="$XCFROOTDIR
echo "ASIROOTDIR="$ASIROOTDIR

#add asi libs to   LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH$ASIROOTDIR/lib/$ARCH

echo "LD_LIBRARY_PATH=" $LD_LIBRARY_PATH

export PYTHONPATH="$XCFROOTDIR/libs/":$PYTHONPATH
echo "pythonpath="$PYTHONPATH
