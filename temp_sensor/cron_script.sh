#!/bin/bash

# script to check if temp sensor scripts are running

A=`ps -ef | grep   app_bokeh | grep -v grep | wc -l`
echo $A

if [ $A = 0 ]; then   
    nohup python /home/xcf/XCF/temp_data/temp_sensor/test2/app_bokeh.py &
    rm /home/xcf/XCF/temp_data/temp_sensor/test2/nohup.out
fi



B=`ps -ef | grep  test_serialRead_nomefile.py  | grep -v grep | wc -l`
echo $B
if [ $B = 0 ]; then   
    nohup python /home/xcf/XCF/temp_data/temp_sensor/test_serialRead_nomefile.py  &
    rm /home/xcf/XCF/temp_data/temp_sensor/nohup.out
fi
