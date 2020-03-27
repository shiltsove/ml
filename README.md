# ml
Intelligent Systems Development

## Description
* MobileNetSSD_deploy.caffemodel - trained model
* MobileNetSSD_deploy.prototxt - ? 
* ssd_evaluation.ipynb -- model evaluation notebook
* TestGround -- ground truth (images descriptions)
* TestImages -- test images


## Install
Install jupyter notebook. (virtual environment optional)
sudo apt update
sudo apt install python3-pip python3-dev
sudo -H pip3 install --upgrade pip
sudo -H pip3 install virtualenv
mkdir ~/my_project_dir
cd ~/my_project_dir
virtualenv my_project_env
source my_project_env/bin/activate

pip install jupyter
jupyter notebook


