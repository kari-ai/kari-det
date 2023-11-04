#!/bin/bash
SOURCE=$1
kari-det predict model=./weights/kari-det.pt source=$SOURCE imgsz=1024 classes=[4,5,6] save=True max_det=1000
