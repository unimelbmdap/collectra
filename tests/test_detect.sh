#!/bin/bash
collectra make -w Hespi -f hespi -d -v 2.0.0
collectra add -w Hespi -t label_detect,object_detection,yolo11n.pt,yolo  
collectra train -w Hespi -t label_detect images/ -o logs --epochs 20