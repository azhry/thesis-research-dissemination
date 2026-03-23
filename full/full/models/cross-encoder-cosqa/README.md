---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:98020
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: accuracy
      value: 0.525
      name: Accuracy
    - type: accuracy_threshold
      value: -4.938349723815918
      name: Accuracy Threshold
    - type: f1
      value: 0.6711590296495957
      name: F1
    - type: f1_threshold
      value: -6.668356895446777
      name: F1 Threshold
    - type: precision
      value: 0.5060975609756098
      name: Precision
    - type: recall
      value: 0.996
      name: Recall
    - type: average_precision
      value: 0.3810564714857781
      name: Average Precision
---

# CrossEncoder based on cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) <!-- at revision 1427fd652930e4ba29e8149678df786c240d8825 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['order of elements in each subset python', 'def get_feature_order(dataset, features):\n    """ Returns a list with the order that features requested appear in\n    dataset """\n    all_features = dataset.get_feature_names()\n\n    i = [all_features.index(f) for f in features]\n\n    return i'],
    ['running python unit tests command line', 'def test():  # pragma: no cover\n    """Execute the unit tests on an installed copy of unyt.\n\n    Note that this function requires pytest to run. If pytest is not\n    installed this function will raise ImportError.\n    """\n    import pytest\n    import os\n\n    pytest.main([os.path.dirname(os.path.abspath(__file__))])'],
    ['how do you kill a program in python', 'def kill(self):\n        """Kill the browser.\n\n        This is useful when the browser is stuck.\n        """\n        if self.process:\n            self.process.kill()\n            self.process.wait()'],
    ['python render sympy latex', 'def print_display_png(o):\n    """\n    A function to display sympy expression using display style LaTeX in PNG.\n    """\n    s = latex(o, mode=\'plain\')\n    s = s.strip(\'$\')\n    # As matplotlib does not support display style, dvipng backend is\n    # used here.\n    png = latex_to_png(\'$$%s$$\' % s, backend=\'dvipng\')\n    return png'],
    ['how to make sprites move up and down in python', 'def move_back(self, dt):\n        """ If called after an update, the sprite can move back\n        """\n        self._position = self._old_position\n        self.rect.topleft = self._position\n        self.feet.midbottom = self.rect.midbottom'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'order of elements in each subset python',
    [
        'def get_feature_order(dataset, features):\n    """ Returns a list with the order that features requested appear in\n    dataset """\n    all_features = dataset.get_feature_names()\n\n    i = [all_features.index(f) for f in features]\n\n    return i',
        'def test():  # pragma: no cover\n    """Execute the unit tests on an installed copy of unyt.\n\n    Note that this function requires pytest to run. If pytest is not\n    installed this function will raise ImportError.\n    """\n    import pytest\n    import os\n\n    pytest.main([os.path.dirname(os.path.abspath(__file__))])',
        'def kill(self):\n        """Kill the browser.\n\n        This is useful when the browser is stuck.\n        """\n        if self.process:\n            self.process.kill()\n            self.process.wait()',
        'def print_display_png(o):\n    """\n    A function to display sympy expression using display style LaTeX in PNG.\n    """\n    s = latex(o, mode=\'plain\')\n    s = s.strip(\'$\')\n    # As matplotlib does not support display style, dvipng backend is\n    # used here.\n    png = latex_to_png(\'$$%s$$\' % s, backend=\'dvipng\')\n    return png',
        'def move_back(self, dt):\n        """ If called after an update, the sprite can move back\n        """\n        self._position = self._old_position\n        self.rect.topleft = self._position\n        self.feet.midbottom = self.rect.midbottom',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.525      |
| accuracy_threshold    | -4.9383    |
| f1                    | 0.6712     |
| f1_threshold          | -6.6684    |
| precision             | 0.5061     |
| recall                | 0.996      |
| **average_precision** | **0.3811** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 98,020 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                     | sentence_1                                                                                         | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                             | float                                                          |
  | details | <ul><li>min: 19 characters</li><li>mean: 37.26 characters</li><li>max: 76 characters</li></ul> | <ul><li>min: 101 characters</li><li>mean: 317.28 characters</li><li>max: 4358 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.09</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                           | sentence_1                                                                                                                                                                                                                                                                                                                                                           | label            |
  |:-----------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>order of elements in each subset python</code> | <code>def get_feature_order(dataset, features):<br>    """ Returns a list with the order that features requested appear in<br>    dataset """<br>    all_features = dataset.get_feature_names()<br><br>    i = [all_features.index(f) for f in features]<br><br>    return i</code>                                                                                  | <code>0.0</code> |
  | <code>running python unit tests command line</code>  | <code>def test():  # pragma: no cover<br>    """Execute the unit tests on an installed copy of unyt.<br><br>    Note that this function requires pytest to run. If pytest is not<br>    installed this function will raise ImportError.<br>    """<br>    import pytest<br>    import os<br><br>    pytest.main([os.path.dirname(os.path.abspath(__file__))])</code> | <code>0.0</code> |
  | <code>how do you kill a program in python</code>     | <code>def kill(self):<br>        """Kill the browser.<br><br>        This is useful when the browser is stuck.<br>        """<br>        if self.process:<br>            self.process.kill()<br>            self.process.wait()</code>                                                                                                                               | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 5
- `fp16`: True
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 5
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss | average_precision |
|:------:|:-----:|:-------------:|:-----------------:|
| 0.0816 | 500   | 0.5528        | -                 |
| 0.1632 | 1000  | 0.3290        | -                 |
| 0.2448 | 1500  | 0.3209        | -                 |
| 0.3264 | 2000  | 0.3054        | -                 |
| 0.4080 | 2500  | 0.3056        | -                 |
| 0.4896 | 3000  | 0.2981        | -                 |
| 0.5712 | 3500  | 0.3095        | -                 |
| 0.6528 | 4000  | 0.3000        | -                 |
| 0.7345 | 4500  | 0.2940        | -                 |
| 0.8161 | 5000  | 0.2983        | -                 |
| 0.8977 | 5500  | 0.3015        | -                 |
| 0.9793 | 6000  | 0.2994        | -                 |
| 1.0    | 6127  | -             | 0.3688            |
| 1.0609 | 6500  | 0.2991        | -                 |
| 1.1425 | 7000  | 0.3063        | -                 |
| 1.2241 | 7500  | 0.3091        | -                 |
| 1.3057 | 8000  | 0.2915        | -                 |
| 1.3873 | 8500  | 0.2934        | -                 |
| 1.4689 | 9000  | 0.2943        | -                 |
| 1.5505 | 9500  | 0.2918        | -                 |
| 1.6321 | 10000 | 0.2934        | -                 |
| 1.7137 | 10500 | 0.2898        | -                 |
| 1.7953 | 11000 | 0.2879        | -                 |
| 1.8769 | 11500 | 0.2934        | -                 |
| 1.9585 | 12000 | 0.2839        | -                 |
| 2.0    | 12254 | -             | 0.3489            |
| 2.0402 | 12500 | 0.2808        | -                 |
| 2.1218 | 13000 | 0.2809        | -                 |
| 2.2034 | 13500 | 0.2861        | -                 |
| 2.2850 | 14000 | 0.2890        | -                 |
| 2.3666 | 14500 | 0.2741        | -                 |
| 2.4482 | 15000 | 0.2627        | -                 |
| 2.5298 | 15500 | 0.2843        | -                 |
| 2.6114 | 16000 | 0.2751        | -                 |
| 2.6930 | 16500 | 0.2863        | -                 |
| 2.7746 | 17000 | 0.2823        | -                 |
| 2.8562 | 17500 | 0.2904        | -                 |
| 2.9378 | 18000 | 0.2820        | -                 |
| 3.0    | 18381 | -             | 0.3716            |
| 3.0194 | 18500 | 0.2756        | -                 |
| 3.1010 | 19000 | 0.2764        | -                 |
| 3.1826 | 19500 | 0.2693        | -                 |
| 3.2642 | 20000 | 0.2758        | -                 |
| 3.3458 | 20500 | 0.2700        | -                 |
| 3.4275 | 21000 | 0.2661        | -                 |
| 3.5091 | 21500 | 0.2722        | -                 |
| 3.5907 | 22000 | 0.2692        | -                 |
| 3.6723 | 22500 | 0.2677        | -                 |
| 3.7539 | 23000 | 0.2548        | -                 |
| 3.8355 | 23500 | 0.2667        | -                 |
| 3.9171 | 24000 | 0.2626        | -                 |
| 3.9987 | 24500 | 0.2681        | -                 |
| 4.0    | 24508 | -             | 0.3572            |
| 4.0803 | 25000 | 0.2658        | -                 |
| 4.1619 | 25500 | 0.2573        | -                 |
| 4.2435 | 26000 | 0.2665        | -                 |
| 4.3251 | 26500 | 0.2593        | -                 |
| 4.4067 | 27000 | 0.2568        | -                 |
| 4.4883 | 27500 | 0.2566        | -                 |
| 4.5699 | 28000 | 0.2458        | -                 |
| 4.6515 | 28500 | 0.2447        | -                 |
| 4.7331 | 29000 | 0.2493        | -                 |
| 4.8148 | 29500 | 0.2580        | -                 |
| 4.8964 | 30000 | 0.2590        | -                 |
| 4.9780 | 30500 | 0.2521        | -                 |
| 5.0    | 30635 | -             | 0.3811            |


### Framework Versions
- Python: 3.13.5
- Sentence Transformers: 5.2.3
- Transformers: 5.3.0
- PyTorch: 2.10.0+cpu
- Accelerate: 1.13.0
- Datasets: 4.6.1
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->