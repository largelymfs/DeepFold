# DeepFold
Learning structural motif representations for efficient protein structure search


## Installation
* Download the pretrained model from https://drive.google.com/open?id=0B8bKX4poTFVta0d6QmswR3dOcms and put it under ./models
* Run `bash setup_env.sh` to install the required packages
* Add the variable `floatX = float32` in your `~/.theanorc` config file under the `[global]` section.

## Usage
The script provides an example of generating an embedding for a single PDB file. PDB file of any length can be used but for maximum efficiency make sure the model size can be fit in the VRAM. The script is not efficient for generating a large number of embeddings due to the overhead. For batch processing, please call the get\_embedding function of a DeepFold instance inside the network module.

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
