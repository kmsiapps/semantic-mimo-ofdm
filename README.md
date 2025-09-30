# Bridging Neural Networks and Wireless Systems with MIMO-OFDM Semantic Communications
[huggingface]: https://huggingface.co/wintersummer01/semantic-mimo-ofdm/tree/main

This repository contains the code for the paper  
[Bridging Neural Networks and Wireless Systems with MIMO-OFDM Semantic Communications](https://arxiv.org/abs/2501.16726).  
Data and model checkpoints for the semantic server are available at [Hugging Face][huggingface].

If you find this repository or code useful, please consider citing our work:
```
@ARTICLE{yoo2025bridging,
  author={Yoo, Hanju and Choi, Dongha and Kim, Yonghwi and Kim, Yoontae and Kim, Songkuk and Chae, Chan-Byoung and Heath, Robert W.},
  journal={IEEE Wireless Communications}, 
  title={Bridging Neural Networks And Wireless Systems with MIMO-OFDM Semantic Communications}, 
  year={2025},
  volume={32},
  number={5},
  pages={48-55},
  month={Sept.}}
```


### Directories
- `SERVER/`: TCP-based server script for semantic encoding/decoding.  
  Requires TensorFlow with CUDA support. Model checkpoints must be downloaded from [Hugging Face][huggingface].
- `CLIENT/`: Baseband transmission scripts.  
  - Sends input images to the server.  
  - Receives encoded symbols, modulates them with OFDM, and transmits via USRP.  
  - Performs reception, ZF equalization, symbol recovery, and decoding through the server.  
  Client PC must be connected to USRPs.

### Requirements
- **Server PC**: TensorFlow, [bpgenc](https://github.com/josejuansanchez/bgp-image-format)  
- **Host PC**: NumPy, SciPy, [Ettus UHD driver](https://files.ettus.com/manual/page_install.html), USRP X310 connection  
- We use Jupyter Notebook extensions in VS Code to run `server.py` / `client.py` with an interactive GUI.

### Run
1. Launch `SERVER/ofdm_docker_server.py` on the server PC.  
2. Configure `CLIENT/tcp_configs.py` with the server and USRP IP/port.  
3. Run:  
   - `CLIENT/ofdm_semantic_client.py` (semantic transmission)  
   - `CLIENT/ofdm_bpg_client.py` (BPG-based transmission)

### Model Training
See `SERVER/run.sh` for training reference.
