# Docker with X forwarding for Jetson Nano: GPIO, Camera, Sound Processing, ML
Purpose : build a reusable image using docker to control GPIO pins, speakers and microphone with ML libraries for Jetson Nano.\
Feel free to modify requirements.txt to install more libraries.

**Installed libraries:**\
TensorFlow 1.15.5 (base image)\
PyTorch v1.9.0 (base image) \
torchvision v0.10.0 (base image)\
torchaudio v0.9.0 (base image) \
onnx 1.8.0 (base image) \
CuPy 9.2.0 (base image)\
numpy 1.19.5 (base image)\
numba 0.53.1 (base image)\
OpenCV 4.5.0 (base image)\
pandas 1.1.5 (base image)\
scipy 1.5.4 (base image)\
scikit-learn 0.23.2 (base image)\
JupyterLab 2.2.9 (base image)\
Jetson.GPIO 2.0.17\
pyaudio 0.2.11\
simpleaudio 1.0.4\
librosa 0.8.1\
transformers 4.15.0

## Installation
### 1) Flash jetson with Jetpack 4.6, no upgrade
### 2) Connect sensors before boot
### 3) Clone official repo for gpio pins control
`cd Desktop`\
`git clone https://github.com/NVIDIA/jetson-gpio.git`
### 4) Clone code for container creation
`git clone https://github.com/nlpTRIZ/container_jetson_audio_gpio.git`\
### 5) Create gpio group
`sudo groupadd -f -r gpio`
### 6) Add user to gpio group
`sudo usermod -a -G gpio $USER`
### 7) Copy the permissions file
`sudo cp jetson-gpio/lib/python/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/`
### 8) Remove GPIO repo
`rm -rf jetson-gpio`
### 9) Add user to docker group
`sudo usermod -a -G docker jetson0`\
### 10) Reboot
`sudo reboot`
### 11) Build image (replace name_image with a proper name)
Create the image of the desired environment from the official image of nvidia in which we execute the contents of Dockerfile.\
Python modules can be added in requirements.txt to add them in the image (check that they are not already there).\
`cd container_jetson_audio_gpio`\
`docker build -t name_image .`
### 12) Load run function
This will permanently set the containers starting command (`drun`) with proper options:\
`. install.sh`\
To temporarly set the command:\
`. copy/drun.sh`
## Run
Once an image is built, no need to rebuid it every time, just start a container.\
Start container (don't forget to connect with -X option: ssh -X user@ip)\
`drun name_image`
