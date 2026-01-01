#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:53:55 2024

@author: zhang
"""
import os
import torch
import pandas as pd
import numpy as np
from scipy import stats
from fairseq.models.roberta import RobertaModel
from tqdm import tqdm

Tasks = {"MRPC":"test.tsv", "RTE": "test.tsv"}

for task in Tasks.keys():
    for eval_file in Tasks[task]:
        model_dir = f'/path/to/saved_models/glue/{task}/'

        roberta = RobertaModel.from_pretrained(
            model_dir,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='/path/to/fairseq-data-bin/glue-roberta/'+task+'-bin'
        )
        positions = {"MRPC":[3, 4], "RTE": [1, 2]}


        label_fn = lambda label: roberta.task.label_dictionary.string(
            [label + roberta.task.label_dictionary.nspecial]
        ) 

        roberta.cuda()
        roberta.eval()
        predictions = []
        with torch.no_grad():
            with open(os.path.join('/path/to/public_data/glue',task, eval_file)) as f:
                data = f.readlines()

            for line in tqdm(data[1:]):
                tokens = line.strip().split('\t')
                
                p1, p2 = positions[task]
                sent1 = tokens[p1]
                if p2 is not None:
                    tokens = roberta.encode(sent1, tokens[p2])
                else:
                    tokens = roberta.encode(sent1)

                prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
                prediction_label = label_fn(prediction)
                predictions.append(prediction_label)

        pred_file = task + ".tsv"
        data = {"index":list(range(len(predictions))), 'label':predictions}
        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(model_dir, pred_file), header=True, index=False, sep="\t")