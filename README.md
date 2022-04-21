# A Fast Post-Training Pruning Framework for Transformers

<div align="center">
  <img src=figures/overview.png>
</div>

## Prerequisite

### Install denpendencies

Tested on Python 3.7.10.

```bash
pip3 install -r requirements.txt
```

### Download checkpoints

We provide the checkpoints of BERT-base and DistilBERT we used in our experiments.
We used the pre-trained weights provided by [HuggingFace Transformers](https://github.com/huggingface/transformers), and fine-tuned them for 8 datasets with standard recipes.

| Model | Link |
|:-----:|:-----:|
| BERT-base | [gdrive](https://drive.google.com/drive/folders/1OWHL7Cjhaf2n67PZX4Pt0Be3Gv2VCLo0?usp=sharing) |
| DistilBERT | [gdrive](https://drive.google.com/drive/folders/1ZyGQL5ynoXs0ffGkENNjHq7eijB-B80l?usp=sharing) |

If you use your own checkpoints, please make sure that each checkpoint directory contains both `config.json` and `pytorch_model.bin`.

## Reproduce the results on GLUE/SQuAD

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
