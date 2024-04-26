import numpy as np
import torch


class MILK_model(torch.nn.Module):
    def __init__(self, env, dataset):
        super(MILK_model, self).__init__()
        self.env = env
        self.n_user = dataset.n_user
        self.m_item = dataset.m_item
        self.free_emb_dimension = self.env.args.free_emb_dimension  # free embedding的维数
        self.feature = torch.tensor(dataset.feature, dtype=torch.float32).to(self.env.device)
        self.feature = torch.nn.functional.normalize(self.feature)
        self.image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        self.image_feat = torch.nn.functional.normalize(self.image_feat)
        self.text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self.text_feat = torch.nn.functional.normalize(self.text_feat)
        if self.env.args.dataset == 'tiktok':
            self.audio_feat = torch.tensor(dataset.audio_feat, dtype=torch.float32).to(self.env.device)
            self.audio_feat = torch.nn.functional.normalize(self.audio_feat)
        self.user_emb = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.free_emb_dimension)

        self.image_linear = torch.nn.Linear(self.image_feat.shape[1], self.free_emb_dimension)
        self.text_linear = torch.nn.Linear(self.text_feat.shape[1], self.free_emb_dimension)
        self.fusion_linear = torch.nn.Linear(self.free_emb_dimension, self.free_emb_dimension)
        self.fusion_linears = torch.nn.ModuleList([torch.nn.Linear(self.free_emb_dimension, self.free_emb_dimension) for i in range(5)])

        self.final_item = None
        self.final_user = None
        self.activate = torch.nn.Sigmoid()

        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        self.to(self.env.device)

    def forward(self, random=False):
        """
        propagate methods for lightGCN
        """
        user_emb = self.user_emb.weight
        mm_emb = (self.image_linear(self.image_feat) + self.text_linear(self.text_feat))/2
        # item_emb = self.fusion_linear(mm_emb)
        item_emb = mm_emb
        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0
        self.final_user = user_emb
        self.final_item = item_emb
        return user_emb, item_emb

    def get_multimedia_emb(self):
        image_emb = self.image_linear(self.image_feat)
        text_emb = self.text_linear(self.text_feat)
        return image_emb, text_emb

    def get_env_emb(self, mix_ration, env):
        """
        propagate methods for lightGCN
        """
        user_emb = self.user_emb.weight
        mm_emb = mix_ration[env][0] * self.image_linear(self.image_feat) + mix_ration[env][1] *  self.text_linear(self.text_feat)#+ mix_ration[env][2] *  self.audio_linear(self.audio_feat)
        # item_emb = self.fusion_linear(mm_emb)
        item_emb = mm_emb
        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0
        return user_emb, item_emb

