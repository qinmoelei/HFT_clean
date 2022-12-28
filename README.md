# HFT
## Environment
  ```
  conda create --name HFT python=3.7.13
  pip install -r requirements.txt
  conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
  ```
## Data
First use the `./tools/compress_trade.py` to compress the trade into OHLCV and follow the `./tools/preprocess.py` to clean and make technical indicator.
Original data could be found [here](https://drive.google.com/drive/folders/1v-rH18d8smqsA0L04mJGWdr4R77vbdjT?usp=sharing).
## Env
The simulation environment could be found in `./env/env_1.py`.
## Algorithm
The core of the algorithm could be found in `./RL/dqn_standard.py`.