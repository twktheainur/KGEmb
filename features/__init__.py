from datasets.kg_dataset import KGDataset
from models import KGModel, torch


class SemanticWebGraphEmbeddingExtractor:
    def __init__(self, dataset: KGDataset, model: KGModel, prefix: str = ""):
        self.dataset = dataset
        self.model = model
        self.prefix = prefix

    def _pref_uri(self, uri):
        return "<" + self.prefix + uri + ">"

    def global_entity_embedding(self, uri: str):
        pass

    def left_hand_side_entity_embedding(self, uri: str):
        id = self.dataset.get_node_id_from_name(self._pref_uri(uri))
        embedding, left_bias = self.model.get_entity_embedding(id, bias="left")
        return embedding * left_bias

    def right_hand_side_entity_embedding(self, uri: str):
        id = self.dataset.get_node_id_from_name(self._pref_uri(uri))
        embedding, right_bias = self.model.get_entity_embedding(id, bias="right")
        return embedding * right_bias

    def relation_embedding(self, uri: str):
        id = self.dataset.get_rel_id_from_name(self._pref_uri(uri))
        return self.model.get_relation_embedding(id)

    def similarity(self, uri_left, uri_rel, uri_right):
        lhs_vector = self.left_hand_side_entity_embedding(uri_left)
        rhs_vector = self.right_hand_side_entity_embedding(uri_right)
        rel_vector = self.relation_embedding(uri_rel)
        return torch.sum(lhs_vector * rel_vector * rhs_vector, 0, keepdim=True).tolist()[0]
