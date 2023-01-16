#!/usr/bin/bash
# @brief How to use defauult arguments
# https://unix.stackexchange.com/questions/25945/how-to-check-if-there-are-no-parameters-provided-to-a-command
OUTPUT=$1
INPUT=$2

if [ "$OUTPUT" == "" ] || ["$INPUT" == ""]
then 
    echo "usage: arg-check OUTPUT INPUT"
    exit 1
else 
    echo $INPUT
    echo $OUTPUT
fi