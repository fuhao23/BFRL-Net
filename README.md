# BFRL-Net
This is a model from the paper "BFRL-Net: Unlocking the Mysteries of Encrypted Traffic".
### How to train
Please first install the runtime environment in requestments.txt. 
```
pip install -r requirements.txt
```
#### PCAP Process
Run the Zeek script to process the traffic file.
```
zeek -Cr filename.pcap FeasExtract.zeek
```
#### Fingerprint feature extraction
Please modify the config.yml file and main.py, then run step 1.
#### Model training
Please modify the config.yml file and main.py, then run step 2.
#### model prediction
Please modify the config.yml file and main.py, then run step 3.
