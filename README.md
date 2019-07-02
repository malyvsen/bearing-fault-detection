# Bearing fault detection
Improving on NASA's work with induction motor bearing fault detection using RNN-powered smart sensors.

## Walkthrough
For starters, you'll want to run `source setup_venv.sh` to automatically setup a Python virtual environment under `bearing_venv`.

Then:
* To preprocess the NASA data, download it from [here](http://data-acoustics.com/measurements/bearing-faults/bearing-4/) and unpack it into `nasa_tests/data`, then `cd nasa_tests` and run `python3 preprocess_data.py`.
* To train the AI, `cd nasa_tests` and run `analysta -vv model single -c lstm_config.json`.
* To view the trained model and a some stats, take a look at [nasa_tests/lstm_results](nasa_tests/lstm_results).
* To view spectrograms of the raw data, head over to [nasa_tests/spectrograms](nasa_tests/spectrograms) - to generate them, you could `cd nasa_tests` and run `python3 spectrograms.py`.
