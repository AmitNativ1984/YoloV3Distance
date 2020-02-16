#!/bin/bash

python YoloV3/data/utils/create_paths_txtfile.py --images-path /home/amit/Data/OID/OID/Dataset/validation/Person_Ambulance_Bus_Truck_Car_Tank_Taxi --dest-txtfile YoloV3/data/datasets/OID_only/Person_Ambulance_Bus_Truck_Car_Tank_Taxi_val.txt
python YoloV3/data/utils/create_paths_txtfile.py --images-path /home/amit/Data/OID/OID/Dataset/train/Person_Ambulance_Bus_Truck_Car_Tank_Taxi --dest-txtfile YoloV3/data/datasets/OID_only/Person_Ambulance_Bus_Truck_Car_Tank_Taxi_train.txt

# merging txt files
# train
python YoloV3/data/utils/merge_txt_files.py --dest YoloV3/data/datasets/OID_Person_allCars_Kitti_Probot_train.txt --txtfile YoloV3/data/datasets/OID_only/Person_Ambulance_Bus_Truck_Car_Tank_Taxi_train.txt YoloV3/data/datasets/kitti_only/Kitti_train.txt  YoloV3/data/datasets/vehicle_recordings_only/vehicle_all_train.txt
# val
python YoloV3/data/utils/merge_txt_files.py --dest YoloV3/data/datasets/OID_Person_allCars_Kitti_Probot_val.txt --txtfile YoloV3/data/datasets/OID_only/Person_Ambulance_Bus_Truck_Car_Tank_Taxi_val.txt YoloV3/data/datasets/kitti_only/Kitti_val.txt  YoloV3/data/datasets/vehicle_recordings_only/vehicle_all_val.txt