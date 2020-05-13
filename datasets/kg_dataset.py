"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl

import numpy as np
import torch


class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2

        self.entity_index = {}
        self.entity_reverse_index = {}
        self.relation_index = {}
        self.relation_reverse_index = {}

        # Loading entity mapping (format: <uri>\t[0-9]+)
        entity_id_file = os.path.join(self.data_path, "ent_id")
        with open(entity_id_file, 'r') as entity_map:
            for line in entity_map.readlines():
                parts = line.split("\t")
                self.entity_index[parts[0]] = int(parts[1])
                self.entity_reverse_index[int(parts[1])] = parts[0]

        # Loading relation mapping (format: <uri>\t[0-9]+)
        rel_id_file = os.path.join(self.data_path, "rel_id")
        with open(rel_id_file, 'r') as rel_map:
            for line in rel_map.readlines():
                parts = line.split("\t")
                self.relation_index[parts[0]] = int(parts[1])
                self.relation_reverse_index[int(parts[1])] = parts[0]

    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities

    def get_node_id_from_name(self, name: str):
        return self.entity_index[name]

    def get_node_name_from_id(self, id: str):
        return self.entity_reverse_index[id]

    def get_rel_id_from_name(self, name: str):
        return self.relation_index[name]

    def get_rel_name_from_id(self, id: str):
        return self.relation_reverse_index[id]
