#!/bin/bash

# Create the required HPSandbox config options file
seq=HHHHH
python generate_conf.py $seq

echo "Working on config having sequence ${seq}"
python enumerate.py "${seq}.conf" > out
mv out "${seq}.conf" 5/

rm -rf 5/anal 5/setup
mv 5/data/by_replica/* 5/
rm -rf 5/data

# Create the configuration from the config options file 
#for k in 5
#do
#    echo "Working on config of length ${k}"
#    cd $k
#    python ../enumerate.py "${k}".conf > out
#    mv "${k}"/data/by_replica/* data
#    rm -rf "${k}"
#    cd ..
#done
