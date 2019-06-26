#!/bin/bash
virtualenv bearing_venv &&\
source bearing_venv/bin/activate &&\
pip install wheel ipykernel tensorflow

if $?
then
    echo "Error during setup, exiting"
    exit
fi

(cd anomaly_detection &&\
pip install -r requirements.txt &&\
python setup.py install)

if $?
then
    echo "Error during setup, exiting"
    exit
fi

ipython kernel install --user --name=bearing_venv

if $?
then
    echo "Error during setup, exiting"
    exit
fi

echo
echo "Finished setting up virtualenv under bearing_venv"
echo "A Jupyter kernel has been installed. To remove it later, run:"
echo "jupyter kernelspec remove bearing_venv"
echo "Remember to deactivate the virtualenv once you're done!"
