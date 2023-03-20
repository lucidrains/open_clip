from src.training.data import get_wds_dataset
from src.training.params import parse_args
import sys
import torch
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss

def dummy(x):
    return x

def main(args):

    args = parse_args(args)
    tokenizer = get_tokenizer(args.model)
    data = get_wds_dataset(args, dummy, is_train=True, tokenizer=tokenizer)
    print(data)
    print(next(iter(data.dataloader)))
    # for batch in data.dataloader:
    #     print(type(batch))
    #     break

if __name__=='__main__':
    main(sys.argv[1:])