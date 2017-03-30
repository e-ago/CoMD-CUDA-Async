#!/bin/bash

cd src-mpi && git pull origin master && make clean && make && cd .. && rm run.log # && sh run.sh &> out.txt
