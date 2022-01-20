"""
Script to run pytorch lightning predict command for embedding papers

"""
import json
import pathlib
import subprocess

import argparse

import logging

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from pytorch_lightning_training_script.train import Specter


class Dataset:

    def __init__(self, data_path, max_length=512, batch_size=32, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        # data is assumed to be a json file
        with open(data_path) as f:
            # key: 'paper_id', value: paper data (including 'title', 'abstract')
            self.data = json.load(f)
        f.close()

    def __len__(self):
        return len(self.data)

    def batches(self):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        i = 0
        for k, d in self.data.items():
            if i % batch_size != 0 or i == 0:
                batch_ids.append(k)
                batch.append(d['title'] + ' ' + (d.get('abstract') or ''))
            else:
                input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                           return_tensors="pt", max_length=self.max_length)
                yield input_ids.to('cuda'), batch_ids
                batch_ids = [k]
                batch = [d['title'] + ' ' + (d.get('abstract') or '')]
            i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=self.max_length)
            input_ids = input_ids.to('cuda')
            yield input_ids, batch_ids


class Model:

    def __init__(self, pl_checkpoint_path, device='cpu'):
        if pl_checkpoint_path is not None:
            # Load the Lightning module first from the checkpoint
            pl_model = Specter.load_from_checkpoint(pl_checkpoint_path)

            # Get the Huggingface BERT model from pl_model
            self.model = pl_model.model
        else:
            self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')

        self.model.to(device)
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output[1]  # cls token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', help='path to the paper ids file to embed')
    parser.add_argument('--model-checkpoint-path', help='path to the model')
    parser.add_argument('--metadata', help='path to the paper metadata')
    parser.add_argument('--output-file', help='path to the output file')
    parser.add_argument('--cuda-device', default=0, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--vocab-dir', default='data/vocab/')
    parser.add_argument('--py_path', default="~/anaconda3/bin/python")
    parser.add_argument('--specter_folder', default=".")

    args = parser.parse_args()

    device = 'cuda' if args.cuda_device == 0 else 'cpu'
    dataset = Dataset(data_path=args.metadata, batch_size=args.batch_size, device=device)
    model = Model(args.model_checkpoint_path, device=device)
    results = {}
    batches = []
    for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // args.batch_size):
        batches.append(batch)
        emb = model(batch)
        for paper_id, embedding in zip(batch_ids, emb.unbind()):
            results[paper_id] = {"paper_id": paper_id, "embedding": embedding.detach().cpu().numpy().tolist()}

    pathlib.Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as fout:
        for res in results.values():
            fout.write(json.dumps(res) + '\n')

    fout.close()


if __name__ == '__main__':
    main()
