import argparse
import os

import torch

from models.bert.model import BertForSequenceClassification
from tools.rewire import rewire_by_magnitude


parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir", type=str)


def main():
    args = parser.parse_args()
    
    model = BertForSequenceClassification.from_pretrained(args.ckpt_dir)
    rewire_by_magnitude(model)

    torch.save(
        model.state_dict(),
        os.path.join(args.ckpt_dir, "pytorch_model.bin"),
    )

if __name__ == "__main__":
    main()
