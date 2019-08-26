# Bearing fault detection
Improving on NASA's work with induction motor bearing fault detection using RNN-powered smart sensors.

## Walkthrough
For starters, you'll want to run `source setup_venv.sh` to automatically setup a Python virtual environment under `bearing_venv`. You may want to experiment with different versions of `analysta` to make sure training works properly.

Then:
* To preprocess the NASA data, download it from [here](http://data-acoustics.com/measurements/bearing-faults/bearing-4/) and unpack it into `bearing-fault-detection/data`, then `cd bearing-fault-detection` and run `python3 preprocess_data.py`.
* To train the AI, `cd bearing-fault-detection` and run `analysta -vv model single -c lstm_config.json`.
* To view the trained model and a some stats, take a look at [bearing-fault-detection/lstm_results](bearing-fault-detection/lstm_results).
* To view spectrograms of the raw data, head over to [bearing-fault-detection/spectrograms](bearing-fault-detection/spectrograms) - to generate them, you could `cd bearing-fault-detection` and run `python3 spectrogram.py`.
