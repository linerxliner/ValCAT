# ValCAT

Source code for our ACL ARR preprint paper:

[ValCAT: Generating Variable-Length Contextualized Adversarial Transformations using Encoder-Decoder](https://openreview.net/forum?id=fE6Md7R_vqA)

(The implement forks [TextAttack](https://github.com/QData/TextAttack) and then refactors it based on our requirements.)

## Dependencies

All required dependencies (Python) are listed in the `requirements.txt` file.

You can install them with this command:

```
pip install -r requirements.txt
```

## Usage

### CLI

Run the following code for textual adversarial attack:

```
python main.py valcat ag-news --result results.json
```

We explain each argument in details:

- attacker: The attack method used. Only `valcat` is supported.
- dataset: The dataset used to generate the adversarial samples. There are six datasets we sampled `ag-news`, `yelp-polarity`, `imdb`, `snli`, `mnli`, `qnli` that can be used.
- --victim: The path to the victim model (It should be a HuggingFace Transformer model). If this argument is not specified, the default victim model of the dataset is automatically used.
- --result: The relative path (relative to the directory where the `main.py` file is located) to save the results of the attack.
- --devices: Two integers that index the two GPUs used for the attack (can point to the same GPU if the GPU memory is large enough), e.g. `--devices 0 1`.
- --dataset_slice: One or two integers are used to slice the dataset. When there is only one integer, it is the start of the slice, e.g. `--dataset_slice 20`. If there are two integers, they are the start and the end of the slice, e.g. `--dataset_slice 20 30`.
- --attack_second_text: For the natural language inference task we attack the two sentences separately. The first sentence is attacked by default, and this argument is applied if we want to attack the second sentence.
- --max_pert_rate: A float indicating the maximum word perturbation rate (constraint) for each adversarial sample.
- --min_use_sim: A float indicating the minimum semantic similarity (constraint) of each adversarial sample.
- --enable_trans: Several integers indicating the applied transformation, e,g, `--enable_trans 1 3`. For `valcat`, `0` indicates an insert operation, `1` indicates a 1-Many replacement operation, `2` indicates a 2-Many replacement operation, and `3` indicates a 3-Many replacement operation.
- --span_order: The largest window size for span importance ranking in ValCAT.
- --encoder_decoder: Encoder-decoder language models for generating adversarial spans. Currently you can use `t5-base` (default), `t5-large`, `t5-v1_1-base` and `mt5-base`.
- --max_candidates: The maximum number of candidate spans that can be generated in each transformation.

### Custom

Our code is modular and allows for custom development.

As an example, after instantiating a `Attacker`, the custom input can be wrapped with `Workload` to generate an adversarial sample:

```
attacker.attack(Workload(text, label=label))
```

