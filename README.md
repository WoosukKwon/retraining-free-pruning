# A Fast Post-Training Pruning Framework for Transformers

Inspired by post-training quantization (PTQ) toolkits, we propose a post-training pruning framework tailored for Transformers.
Different from existing pruning methods, our framework does not require re-training to retain high accuracy after pruning.
This makes our method fully automated and 10x-1000x faster in terms of pruning time.
[[paper link](https://arxiv.org/abs/2204.09656)]

<div align="center">
  <img src=figures/overview.png>
</div>

## Prerequisite

### Install denpendencies

Tested on Python 3.7.10.
You need an NVIDIA GPU (with 16+ GB memory) to run our code.

```bash
pip3 install -r requirements.txt
```

### Prepare checkpoints

We provide the (unpruned) checkpoints of BERT-base and DistilBERT used in our experiments.
We used the pre-trained weights provided by [HuggingFace Transformers](https://github.com/huggingface/transformers), and fine-tuned them for 8 downstream tasks with standard training recipes.

| Model | Link |
|:-----:|:-----:|
| BERT-base | [gdrive](https://drive.google.com/drive/folders/1OWHL7Cjhaf2n67PZX4Pt0Be3Gv2VCLo0?usp=sharing) |
| DistilBERT | [gdrive](https://drive.google.com/drive/folders/1ZyGQL5ynoXs0ffGkENNjHq7eijB-B80l?usp=sharing) |

Our framework only accepts the HuggingFace Transformers PyTorch models.
If you use your own checkpoints, please make sure that each checkpoint directory contains both `config.json` and `pytorch_model.bin`.

## Prune models and test the accuracy on GLUE/SQuAD benchmarks

* Supported models: BERT-base/large, DistilBERT, RoBERTa-base/large, DistilRoBERTa, etc.
* Supported tasks:
  * GLUE: MNLI, QQP, QNLI, SST-2, STS-B, MRPC
  * SQuAD V1.1 & V2

The following example prunes a QQP BERT-base model with 50% MAC (FLOPs) constraint:
```bash
python3 main.py --model_name bert-base-uncased \
                --task_name qqp \
                --ckpt_dir <your HF ckpt directory> \
                --constraint 0.5
```

You can also control more arguments such as sample dataset size (see `main.py`).

## Citation

```bibtex
@misc{kwon2022fast,
      title={A Fast Post-Training Pruning Framework for Transformers}, 
      author={Woosuk Kwon and Sehoon Kim and Michael W. Mahoney and Joseph Hassoun and Kurt Keutzer and Amir Gholami},
      year={2022},
      eprint={2204.09656},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Copyright

THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 02/07/23.
