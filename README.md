# Docker image to control GPIO pins + audio devices using machine learning libraries on Jetson Nano
Purpose : build a reusable image using docker to control GPIO pins, speakers and microphone for Jetson Nano.\
Feel free to modify requirements.txt to install more libraries.

**Installed libraries:**\
TensorFlow 1.15.5 \
PyTorch v1.9.0 \
torchvision v0.10.0\
torchaudio v0.9.0 \
onnx 1.8.0 \
CuPy 9.2.0\
numpy 1.19.5\
numba 0.53.1\
OpenCV 4.5.0 (with CUDA)\
pandas 1.1.5\
scipy 1.5.4\
scikit-learn 0.23.2\
JupyterLab 2.2.9\
Jetson.GPIO 2.0.17\
pyaudio 0.2.11\
simpleaudio 1.0.4\
librosa 0.8.1\
transformers 4.15.0

# Flash jetson with Jetpack 4.6
# brancher capteurs avant boot
cd Documents
# clone code officiel pour le contrôle des pins
git clone https://github.com/NVIDIA/jetson-gpio.git
# clone mon code pour la création d'environnements consistants
git clone https://github.com/nlpTRIZ/container_jetson_audio_gpio.git
sudo apt update
sudo apt install python3-pip
# installation module python pour le contrôle des pins
sudo pip3 install Jetson.GPIO
# création groupe gpio
sudo groupadd -f -r gpio
# ajout utilisateur dans le groupe gpio
sudo usermod -a -G gpio $USER
# on copie le fichier donnant les permissions d'accès dans les règles systèmes
sudo cp jetson-gpio/lib/python/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
# on supprime le code pour le contrôle d'accès car plus besoin
rm -rf jetson-gpio
# on met à jour les permissions système
sudo udevadm control --reload-rules && sudo udevadm trigger
# ajout utilisateur dans le groupe docker pour pouvoir lancer sans sudo
sudo usermod -a -G docker $USER
# on crée l'image de l'environnement souhaité à partir de l'image officielle de nvidia dans laquelle on exécute le contenu du fichier Dockerfile
# des modules pythons peuvent être ajoutés dans requirements.txt pour les installer dans l'image (vérifier qu'ils ne sont pas déjà là de base)
docker build -t jetson_gpio .
# Une fois l'image créée, plus besoin de la recréer, lancer un container à partir de l'image suffit.
# Lancement container
docker run --rm \
	   -it \
	   --runtime=nvidia \
           --net host \
           --gpus all \
           --device /dev/snd \
           --device /dev/bus/usb \
	   --privileged \
	   --cap-add SYS_PTRACE \
	   -e DISPLAY=$DISPLAY \
           -v /sys:/sys \
           -v /tmp/.X11-unix/:/tmp/.X11-unix \
           -v /tmp/argus_socket:/tmp/argus_socket \
           -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
	   -v $(pwd):/app \
           jetson_gpio:latest
