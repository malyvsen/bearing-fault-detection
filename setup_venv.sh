#!/bin/bash
virtualenv bearing_venv &&\
source bearing_venv/bin/activate &&\
pip install wheel ipykernel &&\
pip install -r anomaly_detection/requirements.txt &&\
ipython kernel install --user --name=bearing_venv

echo
echo "Finished setting up virtualenv under bearing_venv"
echo "A Jupyter kernel has been installed. To remove it later, run:"
echo "jupyter kernelspec remove bearing_venv"
echo "Remember to deactivate the virtualenv once you're done!"
