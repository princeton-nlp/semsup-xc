import os
import numpy as np
import pickle
import h5py
from tqdm import tqdm
import sys

# Only two arguments are label file and instance file generated in previous stage

label_file = sys.argv[1]
instance_file = sys.argv[2]



instance_preds = h5py.File(instance_file,'r')

import pickle
import torch
label_preds = pickle.load(open(label_file,'rb'))

def compute_tok_score_cart(doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask):
    qry_input_ids = qry_input_ids.unsqueeze(2).unsqueeze(3)  # Q * LQ * 1 * 1
    doc_input_ids = doc_input_ids.unsqueeze(0).unsqueeze(1)  # 1 * 1 * D * LD
    exact_match = doc_input_ids == qry_input_ids  # Q * LQ * D * LD
    exact_match = exact_match.float()
    scores_no_masking = torch.matmul(
        qry_reps.view(-1, 16),  # (Q * LQ) * d
        doc_reps.view(-1, 16).transpose(0, 1)  # d * (D * LD)
    )
    scores_no_masking = scores_no_masking.view(
        *qry_reps.shape[:2], *doc_reps.shape[:2])  # Q * LQ * D * LD
    scores, _ = (scores_no_masking * exact_match).max(dim=3)  # Q * LQ * D
    tok_scores = (scores * qry_attention_mask.reshape(-1, qry_attention_mask.shape[-1]).unsqueeze(2))[:, 1:].sum(1)
    
    return tok_scores

def semsup_forward(input_embeddings, label_embeddings, num_candidates = -1, list_to_set_mapping = None, same_labels = False):
    print(input_embeddings.shape, label_embeddings.shape)
    logits = torch.bmm(input_embeddings.unsqueeze(1), label_embeddings.transpose(2,1)).squeeze(1)
    return logits

import torch
from typing import Optional
def coil_fast_eval_forward(
    input_ids: Optional[torch.Tensor] = None,
    doc_reps = None,
    logits: Optional[torch.Tensor] = None,
    desc_input_ids = None,
    desc_attention_mask = None,
    lab_reps = None,
    label_embeddings = None
):
    tok_scores = compute_tok_score_cart(
            doc_reps, input_ids,
            lab_reps, desc_input_ids.reshape(-1, desc_input_ids.shape[-1]), desc_attention_mask
    )
    logits = (logits.unsqueeze(0) @ label_embeddings.T)
    new_tok_scores = torch.zeros(logits.shape, device = logits.device)
    for i in range(tok_scores.shape[1]):
        stride = tok_scores.shape[0]//tok_scores.shape[1]
        new_tok_scores[i] = tok_scores[i*stride: i*stride + stride ,i]
    return (logits + new_tok_scores).squeeze()


label_list = [x.strip() for x in open('datasets/Amzn13K/all_labels.txt')]
unseen_label_list = [x.strip() for x in open('datasets/Amzn13K/unseen_labels_split6500_2.txt')]
num_labels = len(label_list)
label_list.sort() # For consistency
l2i = {v: i for i, v in enumerate(label_list)}
unseen_label_indexes = [l2i[x] for x in unseen_label_list]

def get_prediction_results(preds, label_ids):
    top_values = [1,3,5, 10]
    tops = {k:0 for k in top_values}
    for i, (logit, label) in enumerate(zip(preds, label_ids)):
        logit = torch.from_numpy(logit)
        label = torch.from_numpy(label)
        _, indexes = torch.topk(logit, k = max(top_values))
        for val in top_values:
            tops[val] += len([x for x in indexes[:val] if label[x]!=0])
    precisions_at_k =  {k:v/((i+1)*k) for k,v in tops.items()}
    print('Evaluation Result: precision@{} = {}'.format(top_values, precisions_at_k))
    return precisions_at_k

import json
coil_cluster_map = json.load(open('config/bert_coil_map_dict_lemma255K_isotropic.json'))  

all_lab_reps, all_label_embeddings, all_desc_input_ids, all_desc_attention_mask = [], [], [], []
for l in tqdm(label_list):
    ll = label_preds[l]
    lab_reps, label_embeddings, desc_input_ids, desc_attention_mask = ll[np.random.randint(len(ll))]   
    all_lab_reps.append(lab_reps.squeeze())
    all_label_embeddings.append(label_embeddings.squeeze())
    all_desc_input_ids.append(desc_input_ids.squeeze())
    all_desc_attention_mask.append(desc_attention_mask.squeeze())
all_lab_reps = torch.stack(all_lab_reps).cpu()
all_label_embeddings = torch.stack(all_label_embeddings).cpu()
all_desc_input_ids = torch.stack(all_desc_input_ids).cpu()
all_desc_attention_mask = torch.stack(all_desc_attention_mask).cpu()
all_desc_input_ids = torch.tensor([[coil_cluster_map[str(x.item())] for x in xx]  for xx in all_desc_input_ids])

NUM_PTS = 300000
final_labels = np.zeros((NUM_PTS, 13330))
final_logits = np.zeros((NUM_PTS, 13330))
with torch.no_grad():
    for i in tqdm(range(NUM_PTS)):
        item = instance_preds[f'{i}']
        logits, doc_reps, labels, input_ids = torch.tensor(np.array(item['logits'])), torch.tensor(np.array(item['doc_reps'])), torch.tensor(np.array(item['labels'])), torch.tensor(np.array(item['input_ids']))
        final_labels[i] = labels.numpy()
        input_ids = torch.tensor([coil_cluster_map[str(x.item())] for x in input_ids])

        input_ids = input_ids.cuda().unsqueeze(0)
        doc_reps = doc_reps.cuda().unsqueeze(0)
        logits = logits.cuda().unsqueeze(0)
        all_desc_attention_mask = all_desc_attention_mask.cuda()
        all_desc_input_ids = all_desc_input_ids.cuda()
        all_lab_reps = all_lab_reps.cuda()
        all_label_embeddings = all_label_embeddings.cuda()

        final_logits[i] =  coil_fast_eval_forward(input_ids, doc_reps, logits, all_desc_input_ids, all_desc_attention_mask, all_lab_reps, all_label_embeddings).cpu()
        if (i+1)%50000 == 0:
            get_prediction_results(final_logits[:i], final_labels[:i])
            # break

try:
    pickle.dump(final_logits, open(f'final_logits_{sys.argv[3]}.pkl','wb'))
    pickle.dump(final_labels, open(f'final_labels_{sys.argv[3]}.pkl','wb'))
except:
    ...
print('GZSL:')
get_prediction_results(final_logits, final_labels)

unseen_text_indexes = final_labels[:,unseen_label_indexes].sum(1).nonzero()[0]

len(unseen_text_indexes)

final_labels_unseen = final_labels[unseen_text_indexes][:, unseen_label_indexes]
final_logits_unseen = final_logits[unseen_text_indexes][:, unseen_label_indexes]

from tqdm import tqdm
def get_recall_results(preds, label_ids):
    top_values = [1,5,10,20]
    tops = {k:0 for k in top_values}
    for i, (logit, label) in tqdm(enumerate(zip(preds, label_ids))):
        logit = torch.from_numpy(logit)
        label = torch.from_numpy(label)
        _, indexes = torch.topk(logit, k = max(top_values))
        for val in top_values:
            try:
                to_add = (len([x for x in indexes[:val] if label[x]!=0])/label.sum().item())
                tops[val] += to_add
            except:
                ...
    recall_at_k =  {k:v/(i+1) for k,v in tops.items()}
    print('Evaluation Result: recall@{} = {}'.format(top_values, recall_at_k))
    return recall_at_k

get_recall_results(final_logits, final_labels)
print('\nZSL:')
get_prediction_results(final_logits_unseen, final_labels_unseen)

get_recall_results(final_logits_unseen, final_labels_unseen)
