import torch
from engine import Engine
from utils import use_cuda


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.bert_dim = config['bert_dim']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Linear(self.bert_dim, self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()
        
        self.logits = None

    def forward(self, user_indices, item_embeddings):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_embeddings)
        element_product = torch.mul(user_embedding, item_embedding)
        self.logits = self.affine_output(element_product)
        rating = self.logistic(self.logits)
        return rating

    def init_weight(self):
        pass

    def _get_bottleneck(self):
        return self.logits


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)