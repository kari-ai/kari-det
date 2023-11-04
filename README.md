# KARI-Detection (kari-det) - Beta version

Korea Aerospace Research Institute (KARI) AI Lab, 2023.

## Docker Build
```
docker build --tag kari-det:latest .
```
This process may take some time. Please be patient. 
The source code will be authomatically located on /kari-det.

## Docker Run
Run the container with the following command, enabling GPU support, specifying shared memory size, and setting up a volume mount for the data directory:

```
docker run --gpus all -it --shm-size 4g --rm -v images:/kari-det/images:ro kari-det:latest bash
```

## Inference (storage detection)

Activate the 'kari-det' Conda environment using the following command:
```
conda activate kari-det
```
```
cd kari-det
```
If this is your first time running the program, you need to install it first. To do so, enter the following command:
```
pip install -e .
```
To perform inference on your images, execute the following command, replacing the paths with your desired input directories:

```
./kari-det_oil_tanks.sh /kari-det/images
```

If you have any questions, please do not hesitate to contact me at ohhan@kari.re.kr.