from typing import Dict, Optional
import numpy as np
import torch
import itertools
import torch
from torch.utils.data import Dataset
import json
import random
from collections.abc import Mapping
from typing import Dict, Optional, List, Any, NewType
import pandas as pd
from torch.utils.data import DataLoader
from os.path import join
import os
import gensim.downloader
import h5py
import time
from tqdm import tqdm

def getTokenizedLabelDescriptions(data_args, desc_file, tokenizer):
    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.label_max_seq_length, tokenizer.model_max_length)

    label_descs = json.load(open(desc_file, encoding = 'utf-8'))

    return {label_key: [
        tokenizer(
            desc,
            truncation=True,
            padding=padding,
            max_length=max_seq_length,
            return_tensors='pt'
        )
            for desc in descs[1]] for label_key, descs in label_descs.items()}


class SemSupDataset(Dataset):

    def __init__(self, input_dataset, data_args, label_descriptions_file, label_to_id, id_to_label, tokenizer, clsas_descs_len = None, return_desc_embeddings = False, sampleRandom : int = -1, cl_min_positive_descs = 20, useSemSup = True, seen_labels = None, add_label_name = False, max_descs_per_label = 999999, use_precomputed_embeddings = '', bm_short_file = '', ignore_pos_labels_file = '', isTrain = True, class_descs_tokenized = None, choice_indexes = None):
        self.input_dataset = input_dataset
        self.sampleRandom = sampleRandom
        self.cl_min_positive_descs = cl_min_positive_descs
        self.semsup = useSemSup
        self.seen_labels = seen_labels
        self.add_label_name = add_label_name
        self.max_descs_per_label = max_descs_per_label
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.choice_indexes = choice_indexes

        self.bmshortfile = bm_short_file
        self.useBMShort = True if self.bmshortfile!='' else False
        self.data_args = data_args

        self.tok_format = 0
        self.isTrain = isTrain

        # if data_args.large_dset:
            # Instead of loading the 
        self.coil_cluster_map  = None
        try:
            if data_args.coil_cluster_mapping_path:
                self.coil_cluster_map = json.load(open(data_args.coil_cluster_mapping_path))    
        except:
            print('Failed to load cluster map for some reason')
            self.coil_cluster_map  = None
        self.ignore_pos_labels_file = ignore_pos_labels_file
        if self.ignore_pos_labels_file:
            self.ignored_labels = [[y.strip() for y in x.split('\t') if y.strip()!=''] for x in open(self.ignore_pos_labels_file).readlines()]
        else:
            self.ignored_labels = False

        if self.useBMShort and not data_args.large_dset:
            self.shortlists = [[y.strip() for y in x.split('\t')] for x in open(self.bmshortfile).readlines()]

        if self.semsup and not data_args.large_dset:
            self.data_args = data_args
            self.label_descriptions_file = label_descriptions_file
            self.label_to_id = label_to_id
            self.id_to_label = id_to_label
            if self.seen_labels is not None and isinstance(self.seen_labels[0], str):
                self.seen_labels = np.array([self.label_to_id[x] for x in self.seen_labels])
            self.tokenizer = tokenizer
            if class_descs_len is None:
                js_file = json.load(open(self.label_descriptions_file, encoding = 'utf-8'))
                self.class_descs_len = self.tokenize_class_descs(js_file, return_lengths = True)
                self.class_descs = self.tokenize_class_descs(js_file)
            else:
                self.class_descs_len = class_descs_len
            self.return_desc_embeddings = return_desc_embeddings

            self.label_max_seq_length = data_args.label_max_seq_length
            if return_desc_embeddings:
                self.save_tokenized_descs(self.add_label_name)

            if self.use_precomputed_embeddings:
                self.computed_desc_inputs_embeds = torch.from_numpy(np.load(self.use_precomputed_embeddings))
        if self.semsup and data_args.large_dset:
            self.data_args = data_args
            self.label_descriptions_file = label_descriptions_file
            self.label_to_id = label_to_id
            self.id_to_label = id_to_label
            # No concept of seen labels over here, directly load the shortlists
            self.tokenizer = tokenizer
            self.return_desc_embeddings = return_desc_embeddings
            self.label_max_seq_length = data_args.label_max_seq_length

            to_save = True
            if os.path.exists(data_args.tokenized_descs_file):
                print('Path Exists')
                if data_args.tok_format == 1:
                    self.tok_format = 1
                if class_descs_tokenized is not None:
                    self.class_descs_tokenized = class_descs_tokenized
                else:
                    if data_args.tokenized_descs_file.endswith('h5'):
                        self.class_descs_tokenized = h5py.File(data_args.tokenized_descs_file) # np.load(data_args.tokenized_descs_file, allow_pickle=True).item()
                        self.tok_format = 1
                    else:
                        self.class_descs_tokenized = np.load(data_args.tokenized_descs_file, allow_pickle=True)

                # TODO: Fix this hardcoding
                # if len(arr) < int(1e6):
                #     to_save = True # Possibly Corrupt File
                #     # All set, load the file
                # else:
                to_save = False
            js_file = json.load(open(self.label_descriptions_file, encoding = 'utf-8'))
            print('Loaded js File')
            self.class_descs_len = self.tokenize_class_descs(js_file, return_lengths = True)
            if to_save:
                self.class_descs = self.tokenize_class_descs(js_file)
                print('Begin Tokenization Process')
                self.save_tokenized_descs(self.add_label_name)
                print('Saving Tokenized Descriptions')
                import pickle
                pickle.dump(self.class_descs_tokenized, open(data_args.tokenized_descs_file,'wb'))
                print(len(self.class_descs_tokenized))
                3/0
                file = h5py.File(data_args.tokenized_descs_file,'w')
                for key in tqdm(self.class_descs_tokenized):
                    key_h5 = key
                    if key.find('/') != -1:
                        print('There may be issue with', key)
                        key_h5 = key.replace('/','\/')
                    file.create_dataset(key_h5+'/'+'input_ids', data = np.array(self.class_descs_tokenized[key]['input_ids']))
                    file[key_h5].create_dataset('attention_mask', data = np.array(self.class_descs_tokenized[key]['attention_mask']))
            # else:
            #     self.class_descs_tokenized = np.load(data_args.tokenized_descs_file).item()

            if isTrain:
                self.shortlists = h5py.File(data_args.train_tfidf_short)['data']
            else:
                print('Testtt File Loaded')
                self.shortlists = h5py.File(data_args.test_tfidf_short)['data']

        try:
            del self.class_descs
        except: ...
        if self.tok_format != 1:
            self.class_descs_tokenized = pd.DataFrame({k: [np.array(x) for i, x in enumerate(v.values()) if i != 1] for k,v in self.class_descs_tokenized.items()})

    def tokenize_class_descs(self, label_descs, return_lengths = False):
        if return_lengths == 1:
            return {
                label_key: min(descs[0],self.max_descs_per_label)  for label_key, descs in label_descs.items() 
            } # descs 0 is the length
        else:
            return {
                label_key: descs[1][:self.max_descs_per_label] for label_key, descs in label_descs.items() 
            }
            
    def save_tokenized_descs(self, add_label_name = False):
        self.class_descs_tokenized = dict()
        for label_key in tqdm(list(self.class_descs.keys())):
            descs_len = self.class_descs_len[label_key]
            descs = self.class_descs[label_key]
            self.class_descs_tokenized[label_key] = self.tokenizer(
                [label_key + ". " + x for x in descs] if add_label_name else
                descs,
                max_length = self.label_max_seq_length, padding = 'max_length', truncation= True)
            # del self.class_descs_tokenized[label_key]['token_type_ids']

    def __len__(self):
        return len(self.input_dataset)
    

    def get_item_for_large_dset(self, idx, item):
        if self.choice_indexes is not None:
            idx = int(self.choice_indexes[idx]) 
            # print(idx)
        shortlists = self.shortlists[idx]
        labels_new = item['label']
        
        if self.sampleRandom != -1:
            if self.sampleRandom < len(shortlists):
                shortlists = np.random.choice(shortlists, self.sampleRandom, replace = False)
            elif self.sampleRandom > len(shortlists):
                # randomly choose from all remaining labels
                shortlists = shortlists.tolist() + [self.label_to_id[x] for x in np.random.choice(self.seen_labels, self.sampleRandom - len(shortlists), replace = False)]
        if self.isTrain:
            pos_labels = np.where(np.array(labels_new) == 1)[0]
            item['all_candidate_labels'] = np.unique(np.concatenate([pos_labels, shortlists]))[:len(shortlists)]
        else:
            item['all_candidate_labels'] = np.unique(shortlists)
        if self.sampleRandom!=-1:
            if len(item['all_candidate_labels']) < self.sampleRandom:
                # Duplicate entries were deleted, manually add some duplicates :)
                item['all_candidate_labels'] = np.concatenate([item['all_candidate_labels'], item['all_candidate_labels'][len(item['all_candidate_labels'])-self.sampleRandom:]])

            item['all_candidate_labels'] = item['all_candidate_labels'][:self.sampleRandom]
        
        l1 = len(item['all_candidate_labels'])
        if self.ignored_labels:
            # Remove the ignored labels
            # After removing make sure the size is equal to l1, by randomly duplicating elements
            ignore_list = {self.label_to_id[x] for x in self.ignored_labels}
            if len(ignore_list) > 0:
                item['all_candidate_labels'] = set(item['all_candidate_labels'].tolist()).difference(ignore_list)
                item['all_candidate_labels'] = sorted(list(item['all_candidate_labels']))

            if len(item['all_candidate_labels']) < l:
                item['all_candidate_labels'] += item['all_candidate_labels'][:l - len(item['all_candidate_labels'])] 
            item['all_candidate_labels'] = np.array(item['all_candidate_labels'])

        # l1 = np.array(item['label']).sum()
        item['label'] = np.array(item['label'])[item['all_candidate_labels']] 
        # print(f'{item["label"].sum()} / {l1}')
        item['label_desc_ids'] = [np.random.randint(0, self.class_descs_len[self.id_to_label[label_key]]) for label_key in item['all_candidate_labels']]
        
        if self.tok_format ==1:
            item['desc_input_ids'] = [self.class_descs_tokenized['input_ids'][label_key][item['label_desc_ids'][i]].astype(np.int32) for i, label_key in enumerate(item['all_candidate_labels'])]
            item['desc_attention_mask'] = [self.class_descs_tokenized['attention_mask'][label_key][item['label_desc_ids'][i]].astype(np.int32) for i, label_key in enumerate(item['all_candidate_labels'])]
        else:
            item['desc_input_ids'] = [self.class_descs_tokenized[self.id_to_label[label_key]][0][item['label_desc_ids'][i]] for i, label_key in enumerate(item['all_candidate_labels'])]
            item['desc_attention_mask'] = [self.class_descs_tokenized[self.id_to_label[label_key]][1][item['label_desc_ids'][i]] for i, label_key in enumerate(item['all_candidate_labels'])]
        pos_pts = item['label'].nonzero()[0]
        # if len(pos_pts) > 0:
        #     print(idx, item['desc_input_ids'][pos_pts[0]])

        if self.coil_cluster_map:
            map_to_cluster = lambda x : self.coil_cluster_map[str(x)]
            if isinstance(item['input_ids'], list):
                item['clustered_input_ids'] = [self.coil_cluster_map[str(x)] for x in item['input_ids']]
            else:
                item['clustered_input_ids'] = item['input_ids'].vectorize(map_to_cluster)
            item['clustered_desc_ids'] = [[self.coil_cluster_map[str(x)] for x in xx]  for xx in item['desc_input_ids']]

        return item

    def __getitem__(self, idx):
        item = self.input_dataset.__getitem__(idx)
        if self.data_args.large_dset:
            return self.get_item_for_large_dset(idx, item)
        

        # Iterate over all the labels of input_dataset
        # and add random label_description to the item in the same order
        if self.ignored_labels:
            ignored_labels = self.ignored_labels[idx]
        if self.sampleRandom != -1:
            # Create all_candidate_labels
            if self.seen_labels is None:
                labels_new = item['label']
            else:
                labels_new = np.array(item['label'])[self.seen_labels]

            if self.useBMShort:
                # Instead of choosing randomly, choose 60% topmost most from the shortlist
                # Next sample the remaining random entries 
                if self.seen_labels is not None:
                    # from pdb import set_trace as bp
                    # bp()
                    all_candidate_labels = [self.seen_labels.tolist().index(self.label_to_id[x]) for x in self.shortlists[idx] if self.label_to_id[x] in self.seen_labels][:int(0.8*self.sampleRandom)] 
                    # print(f'BM got: {len(all_candidate_labels)}')
                    # Choose the remaining randomly from set of seen_labels - all_candidates
                    all_candidate_labels += np.random.choice(list({x for x in range(len(self.seen_labels))}.difference(set(all_candidate_labels))), self.sampleRandom - len(all_candidate_labels), replace = False).tolist()
            else:
                all_candidate_labels = np.random.choice(range(len(labels_new)) , self.sampleRandom , replace = False)
            # prepend positive labels
            pos_labels = np.where(np.array(labels_new) == 1)[0]
            all_candidate_labels = np.concatenate([pos_labels, all_candidate_labels])
            # Remove duplicates
            all_candidate_labels = np.unique(all_candidate_labels)[:self.sampleRandom]
            if len(pos_labels) < self.cl_min_positive_descs:
                addn_pos_labels = np.random.choice(pos_labels, self.cl_min_positive_descs - len(pos_labels))
                all_candidate_labels = np.concatenate([addn_pos_labels, all_candidate_labels])[:self.sampleRandom]
            np.random.shuffle(all_candidate_labels)
            item['all_candidate_labels'] = all_candidate_labels
            # NOTE: ids will be according to seen labels
            # Now update the labels based on all_candidate_labels

        # print('Getting Data')  
        if self.semsup:
            # print(len(item['label']))
            if 'all_candidate_labels' not in item:
                item['label_desc_ids'] = [np.random.randint(0, self.class_descs_len[self.id_to_label[label_key]]) for label_key in range(len(item['label']))]
                if self.return_desc_embeddings:
                    item['desc_input_ids'] = [self.class_descs_tokenized[self.id_to_label[label_key]][0][item['label_desc_ids'][label_key]] for label_key in range(len(item['label']))]
                    item['desc_attention_mask'] = [self.class_descs_tokenized[self.id_to_label[label_key]][1][item['label_desc_ids'][label_key]] for label_key in range(len(item['label']))]
                    if self.use_precomputed_embeddings:
                        new_indices = [i*5 + x for i,x in enumerate(item['label_desc_ids'])]
                        # item['desc_inputs_embeds'] = [self.computed_desc_inputs_embeds[ item['label_desc_ids'][label_key], self.label_to_id[self.id_to_label[label_key]] ] for label_key in range(len(item['label']))]
                        # item['desc_inputs_embeds'] = self.computed_desc_inputs_embeds[ item['label_desc_ids'][label_key], self.label_to_id[self.id_to_label[label_key]]  for label_key in range(len(item['label']))]
                        if self.seen_labels is not None:
                            new_indices = [x for i, x in enumerate(new_indices) if i in self.seen_labels]
                        item['desc_inputs_embeds'] = self.computed_desc_inputs_embeds[new_indices]

                item['all_candidate_labels'] = range(len(item['label']))

                if self.seen_labels is not None:
                    item['label_desc_ids'] = (np.array(item['label_desc_ids'])[self.seen_labels]).tolist()
                    if self.return_desc_embeddings:
                        item['desc_input_ids'] = (np.array(item['desc_input_ids']))[self.seen_labels].tolist()
                        item['desc_attention_mask'] = (np.array(item['desc_attention_mask']))[self.seen_labels].tolist()
                        # if self.use_precomputed_embeddings:
                        #     item['desc_inputs_embeds'] = torch.tensor(item['desc_inputs_embeds'])[self.seen_labels]

                    item['all_candidate_labels'] = (np.array(item['all_candidate_labels']))[self.seen_labels].tolist()
                    item['label'] = (np.array(item['label']))[self.seen_labels].tolist()
            elif 'all_candidate_labels' in item:
                # print('Computing')
                st = time.time()
                item['label_desc_ids'] = [np.random.randint(0, self.class_descs_len[self.id_to_label[label_key]]) for label_key in range(len(item['label']))]
                if self.seen_labels is not None:
                    if self.return_desc_embeddings:
                        item['desc_input_ids'] = [self.class_descs_tokenized[self.id_to_label[label_key]][0][item['label_desc_ids'][label_key]] for label_key in range(len(item['label']))]
                        item['desc_attention_mask'] = [self.class_descs_tokenized[self.id_to_label[label_key]][1][item['label_desc_ids'][label_key]] for label_key in range(len(item['label']))]
                        if self.use_precomputed_embeddings:
                            new_indices = [i*5 + x for i,x in enumerate(item['label_desc_ids'])]
                            # Now of the 4271 labels, chose only the seen labels
                            new_indices = [x for i, x in enumerate(new_indices) if i in self.seen_labels]
                            # Now choose all_candidate labels
                            # print(len(new_indices))
                            new_indices = [new_indices[x] for x in sorted(item['all_candidate_labels'])]
                            # print(len(new_indices), len(item['all_candidate_labels']))
                            # if len(new_indices)!=1500:
                            #     print('Some Issue Over Here')
                            item['desc_inputs_embeds'] = self.computed_desc_inputs_embeds[new_indices]
                                # [self.computed_desc_inputs_embeds[ item['label_desc_ids'][label_key], self.label_to_id[self.id_to_label[label_key]] ] for label_key in range(len(item['label']))]
                        # print('Mid Calculation Done', item['desc_inputs_embeds'].shape, time.time() - st)
                    item['label_desc_ids'] = np.array(item['label_desc_ids'])[self.seen_labels].tolist()
                    item['label'] = np.array(item['label'])[self.seen_labels].tolist()
                    item['label'] = np.array(item['label'])[all_candidate_labels].tolist()
                    item['desc_input_ids'] = np.array(item['desc_input_ids'])[self.seen_labels][item['all_candidate_labels']].tolist()
                    item['desc_attention_mask'] = np.array(item['desc_attention_mask'])[self.seen_labels][item['all_candidate_labels']].tolist()
                    # if self.use_precomputed_embeddings:
                        # print('Starting Final Compute', time.time() - st)
                        # item['desc_inputs_embeds'] = item['desc_inputs_embeds'][self.seen_labels][item['all_candidate_labels']]#.tolist()
                    # print('Computed', type(item['desc_inputs_embeds']), type(item['desc_inputs_embeds'][0]), time.time() - st)
                else:
                    item['label'] = np.array(item['label'])[all_candidate_labels].tolist()
                    if self.return_desc_embeddings:
                        item['desc_input_ids'] = [self.class_descs_tokenized[self.id_to_label[label_key]][0][item['label_desc_ids'][label_key]] for label_key in np.array(item['all_candidate_labels'])]
                        item['desc_attention_mask'] = [self.class_descs_tokenized[self.id_to_label[label_key]][1][item['label_desc_ids'][label_key]] for label_key in np.array(item['all_candidate_labels'])]
                        if self.use_precomputed_embeddings:
                            item['desc_inputs_embeds'] = [self.computed_desc_inputs_embeds[ item['label_desc_ids'][label_key], self.label_to_id[self.id_to_label[label_key]] ] for label_key in np.array(item['all_candidate_labels'])]

            if self.ignored_labels:
                if self.sampleRandom != -1 and self.seen_labels is not None:
                    ignored_labels = [self.seen_labels.tolist().index(self.label_to_id[x]) for x in self.ignored_labels[idx]]
                    item['all_candidate_labels'] = item['all_candidate_labels'].tolist()
                else:
                    ignored_labels = [self.label_to_id[x] for x in self.ignored_labels[idx]]
                remove_pts = [item['all_candidate_labels'].index(x) for x in ignored_labels if x in item['all_candidate_labels']]
                keep_pts = [x for x in range(len(item['all_candidate_labels'])) if x not in remove_pts]
                # Keep pts can be less than sampleRandom. Manually pad after choosing some values
                # print('Before Len', len(keep_pts), len(item['desc_input_ids']))
                if self.sampleRandom!=-1 and len(keep_pts) < self.sampleRandom:
                    # print('Inside the choice function')
                    keep_pts += np.random.choice(keep_pts, self.sampleRandom - len(keep_pts), replace = False).tolist()
                # print('After Len', len(keep_pts), len(item['desc_input_ids']))
                
                # print(len(keep_pts), max(keep_pts))
                item['desc_input_ids'] = np.array(item['desc_input_ids'])[keep_pts].tolist()
                item['desc_attention_mask'] = np.array(item['desc_attention_mask'])[keep_pts].tolist()
                if 'desc_inputs_embeds' in item:
                    item['desc_inputs_embeds'] = np.array(item['desc_inputs_embeds'])[keep_pts].tolist()
                item['label_desc_ids'] = np.array(item['label_desc_ids'])[keep_pts].tolist()
                item['label'] = np.array(item['label'])[keep_pts].tolist()

            if self.coil_cluster_map:
                map_to_cluster = lambda x : self.coil_cluster_map[str(x)]
                if isinstance(item['input_ids'], list):
                    item['clustered_input_ids'] = [self.coil_cluster_map[str(x)] for x in item['input_ids']]
                else:
                    item['clustered_input_ids'] = item['input_ids'].vectorize(map_to_cluster)
                item['clustered_desc_ids'] = [[self.coil_cluster_map[str(x)] for x in xx]  for xx in item['desc_input_ids']]
            return item


        else:
            return item


