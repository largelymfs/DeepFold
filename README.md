# DeepFold
Learning structural motif representations for efficient protein structure search


## Installation
* Download the pretrained model from https://drive.google.com/open?id=0B8bKX4poTFVta0d6QmswR3dOcms and put it under ./models
* Run `bash setup_env.sh` to install the required packages
* Add the variable `floatX = float32` in your `~/.theanorc` config file under the `[global]` section.

## Usage
```
python ./scripts/gen_embedding.py [-h] [--model model] pdb_file output_file

Create embeddings for a protein structure. Output a numpy embedding.

positional arguments:
  pdb_file       an input pdb file
    output_file    an output numpy embedding

    optional arguments:
      -h, --help     show this help message and exit
        --model model  the network model to load
```


