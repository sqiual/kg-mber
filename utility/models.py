import numpy as np
from numpy import random
from random import random
import pickle
import scipy.sparse as sp
import datetime
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.utils.data as dataloader
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from config.configurator import configs
from utility.parser import parse_args
args = parse_args()

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
torch.autograd.set_detect_anomaly(True)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class Augmentor():
    def __init__(self, data_config, args):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.train_matrix = data_config['trn_mat']
        self.training_user, self.training_item = self.get_train_interactions()
        self.ssl_ratio = args.ssl_ratio
        self.aug_type = args.aug_type

    def get_train_interactions(self):
        users_list, items_list = [], []
        for (user, item), value in self.train_matrix.items():
            users_list.append(user)
            items_list.append(item)

        return users_list, items_list

    def augment_adj_mat(self, aug_type=0):
        np.seterr(divide='ignore')
        n_nodes = self.n_users + self.n_items
        if aug_type in [0, 1, 2] and self.ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_user_idx = np.random.choice(self.n_users, size=int(self.n_users * self.ssl_ratio),
                                                 replace=False)
                drop_item_idx = np.random.choice(self.n_items, size=int(self.n_items * self.ssl_ratio),
                                                 replace=False)
                indicator_user = np.ones(self.n_users, dtype=np.float32)
                indicator_item = np.ones(self.n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)  # [n_user, n_user]
                diag_indicator_item = sp.diags(indicator_item)  # [n_item, n_item]
                R = sp.csr_matrix(
                    (np.ones_like(self.training_user, dtype=np.float32), (self.training_user, self.training_item)),
                    shape=(self.n_users, self.n_items))
                R_prime = diag_indicator_user.dot(R).dot(
                    diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.n_users)),
                                        shape=(n_nodes, n_nodes))
            if aug_type in [1, 2]:
                keep_idx = np.random.choice(len(self.training_user),
                                            size=int(len(self.training_user) * (1 - self.ssl_ratio)),
                                            replace=False)
                user_np = np.array(self.training_user)[keep_idx]
                item_np = np.array(self.training_item)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)), shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # print(adj_matrix.tocsr())
        return adj_matrix.tocsr()

class KGMBR(nn.Module):
    name = 'KGMBR'

    def __init__(self, max_item_list, data_config, args):
        super(KGMBR, self).__init__()
        # ********************** input data *********************** #
        self.max_item_list = max_item_list
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.num_nodes = self.n_users + self.n_items
        self.pre_adjs = data_config['pre_adjs']
        self.pre_adjs_tensor = [self._convert_sp_mat_to_sp_tensor(adj).to(configs['device']) for adj in self.pre_adjs]
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        # ********************** hyper parameters *********************** #
        self.coefficient = torch.tensor(eval(args.coefficient)).view(1, -1).to(configs['device'])
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)  # dropout ratio
        self.aug_type = args.aug_type
        self.nhead = args.nhead
        self.att_dim = args.att_dim
        # ********************** learnable parameters *********************** #
        self.all_weights = {}
        self.all_weights['user_embedding'] = Parameter(torch.FloatTensor(self.n_users, self.emb_dim))
        self.all_weights['item_embedding'] = Parameter(torch.FloatTensor(self.n_items, self.emb_dim))
        self.all_weights['relation_embedding'] = Parameter(torch.FloatTensor(self.n_relations, self.emb_dim))

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            self.all_weights['W_gc_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))
            self.all_weights['W_rel_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))

        self.all_weights['trans_weights_s1'] = Parameter(
            torch.FloatTensor(self.n_relations, self.emb_dim, self.att_dim))
        self.all_weights['trans_weights_s2'] = Parameter(torch.FloatTensor(self.n_relations, self.att_dim, 1))
        self.reset_parameters()
        self.all_weights = nn.ParameterDict(self.all_weights)
        self.dropout = nn.Dropout(self.mess_dropout[0], inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.all_weights['user_embedding'])
        nn.init.xavier_uniform_(self.all_weights['item_embedding'])
        nn.init.xavier_uniform_(self.all_weights['relation_embedding'])
        nn.init.xavier_uniform_(self.all_weights['trans_weights_s1'])
        nn.init.xavier_uniform_(self.all_weights['trans_weights_s2'])
        for k in range(self.n_layers):
            nn.init.xavier_uniform_(self.all_weights['W_gc_%d' % k])
            nn.init.xavier_uniform_(self.all_weights['W_rel_%d' % k])

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape))

    def kg_init_transE(self, kgdataset, recommend_model, kg_optimizer, index):
            Recmodel = recommend_model
            Recmodel.train()
            kgloader = DataLoader(kgdataset, batch_size=4096, drop_last=True)
            opt = kg_optimizer
            trans_loss = 0.
            for data in tqdm(kgloader, total=len(kgloader), disable=True):
                heads = data[0].to(configs['device']) 
                relations = data[1].to(configs['device'])
                pos_tails = data[2].to(configs['device'])
                neg_tails = data[3].to(configs['device'])
                kg_batch_loss = Recmodel.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails, index)
                trans_loss += kg_batch_loss / len(kgloader)
                opt.zero_grad()  #
                kg_batch_loss.backward() #
                opt.step() #
            return trans_loss.cpu().item()

    def kg_init_transR(self, kgdataset, recommend_model, kg_optimizer, index):
            Recmodel = recommend_model
            Recmodel.train()
            kgloader = DataLoader(kgdataset, batch_size=4096, drop_last=True)
            opt = kg_optimizer
            trans_loss = 0.
            for data in tqdm(kgloader, total=len(kgloader), disable=True):
                heads = data[0].to(configs['device']) # 
                relations = data[1].to(configs['device'])
                pos_tails = data[2].to(configs['device'])
                neg_tails = data[3].to(configs['device'])
                kg_batch_loss = Recmodel.calc_kg_loss_transR(heads, relations, pos_tails, neg_tails, index)
                trans_loss += kg_batch_loss / len(kgloader)
                opt.zero_grad()  #
                kg_batch_loss.backward() #
                opt.step() #
            return trans_loss.cpu().item()


    def kg_init_TATEC(self, kgdataset, recommend_model, opt, index):
        Recmodel = recommend_model #KGModel
        Recmodel.train()
        kgloader = DataLoader(kgdataset, batch_size=4096, drop_last=True)
        trans_loss = 0.
        for data in tqdm(kgloader, total=len(kgloader), disable=True):
            heads = data[0].to(configs['device']) #
            relations = data[1].to(configs['device']) 
            pos_tails = data[2].to(configs['device']) 
            neg_tails = data[3].to(configs['device']) 
            kg_batch_loss = Recmodel.calc_kg_loss_TATEC(heads, relations, pos_tails, neg_tails, index)
            trans_loss += kg_batch_loss / len(kgloader)
            opt.zero_grad()
            kg_batch_loss.backward()
            opt.step()
        return trans_loss.cpu().item()


    def BPR_train_contrast(self, u_batch_list, i_batch_list, ua_embeddings, ia_embeddings, Kg_model, contrast_model, contrast_views, ssl_reg=0.1):
        Recmodel = Kg_model
        uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
        # uiv1 = uiv1.coalesce()  # Make sure it's a coalesced sparse tensor
        # uiv1.values().fill_(1) 
        # uiv2 = uiv2.coalesce()  # Make sure it's a coalesced sparse tensor
        # uiv2.values().fill_(1) 
        l_ssl = list()
        items = i_batch_list

        usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, ua_embeddings[:, -1, :], ia_embeddings[:, -1, :], index=0) # contain target-embedding
        usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, ua_embeddings[:, -1, :], ia_embeddings[:, -1, :], index=1)
        # usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, index=0) # 
        # usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, index=1)
        items_uiv1 = itemsv1_ro[items] #
        items_uiv2 = itemsv2_ro[items]
        l_item = contrast_model.grace_loss(items_uiv1, items_uiv2) #KCL 

        users = u_batch_list #
        users_uiv1 = usersv1_ro[users] # 
        users_uiv2 = usersv2_ro[users]
        l_user = contrast_model.grace_loss(users_uiv1, users_uiv2) # 
        l_ssl.extend([l_user * ssl_reg, l_item * ssl_reg]) # 
        
        l_ssl = torch.stack(l_ssl).sum() #

        return l_ssl

    def forward(self, device):      
        ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']),
                                   dim=0).unsqueeze(1).repeat(1, self.n_relations, 1)

        all_embeddings = ego_embeddings        
        all_rela_embs = {}
        
        for i in range(self.n_relations):
            beh = self.behs[i]
            rela_emb = self.all_weights['relation_embedding'][i]
            rela_emb = torch.reshape(rela_emb, (-1, self.emb_dim))
            all_rela_embs[beh] = [rela_emb]

        total_mm_time = 0.
        for k in range(0, self.n_layers):
            embeddings_list = []
            for i in range(self.n_relations):
                # st = time()
                embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings[:, i, :])
                # total_mm_time += time() - st
                rela_emb = all_rela_embs[self.behs[i]][k]
                embeddings_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
                embeddings_list.append(embeddings_)
            embeddings_st = torch.stack(embeddings_list, dim=1)
            embeddings_list = []
            attention_list = []

            for i in range(self.n_relations):
                attention = F.softmax(
                    torch.matmul(
                        torch.tanh(torch.matmul(embeddings_st, self.all_weights['trans_weights_s1'][i])),
                        self.all_weights['trans_weights_s2'][i]
                    ).squeeze(2),
                    dim=1
                ).unsqueeze(1)
                attention_list.append(attention)
                embs_cur_rela = torch.matmul(attention, embeddings_st).squeeze(1)
                #embs_cur_rela = torch.mean(embeddings_st, dim = 1)
                embeddings_list.append(embs_cur_rela)
            ego_embeddings = torch.stack(embeddings_list, dim=1)
            attn = torch.cat(attention_list, dim=1)
            ego_embeddings = self.dropout(ego_embeddings)
            all_embeddings = all_embeddings + ego_embeddings

            for i in range(self.n_relations):
                rela_emb = torch.matmul(all_rela_embs[self.behs[i]][k],
                                        self.all_weights['W_rel_%d' % k])
                all_rela_embs[self.behs[i]].append(rela_emb)

        all_embeddings /= self.n_layers + 1
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        io_g_embeddings = i_g_embeddings
        token_embedding = torch.zeros([1, self.n_relations, self.emb_dim], device=device)
        i_g_embeddings = torch.cat((i_g_embeddings, token_embedding), dim=0)
        attn_user, attn_item = torch.split(attn, [self.n_users, self.n_items], 0)

        for i in range(self.n_relations):
            all_rela_embs[self.behs[i]] = torch.mean(torch.stack(all_rela_embs[self.behs[i]], 0), 0)

        return u_g_embeddings, i_g_embeddings, io_g_embeddings, all_rela_embs, attn_user, attn_item



def drop_edge_random(item2entities, p_drop, padding): 
    res = dict()
    for item, es in item2entities.items():
        new_es = list()
        for e in es.tolist():
            if (random() > p_drop): 
                new_es.append(e) 
            else:
                new_es.append(padding) 
        res[item] = torch.IntTensor(new_es).to(configs['device']) 
    return res 


class Contrast(nn.Module): # KG-contrast
    def __init__(self, Kg_model, tau):
        super(Contrast, self).__init__()
        self.Kg_model = Kg_model
        self.tau = tau #0.2

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]: 
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1) 
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t()) 

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.pair_sim(z1, z2)) 
        return torch.sum(-torch.log(between_sim.diag() / (between_sim.sum(1) - between_sim.diag()))) 

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = False, batch_size: int = 0):
        h1 = z1
        h2 = z2
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            # l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        return l1 #

    def get_kg_views(self):
        
        kg = self.Kg_model.kg_dict 
        view1 = drop_edge_random(kg, configs['model']['kg_p_drop'], 
                                 self.Kg_model.num_entities)
        view2 = drop_edge_random(kg, configs['model']['kg_p_drop'],
                                 self.Kg_model.num_entities)
        return view1, view2

    def item_kg_stability(self, view1, view2, index):  
        kgv1_ro = self.Kg_model.cal_item_embedding_from_kg(view1, index=index) 
        kgv2_ro = self.Kg_model.cal_item_embedding_from_kg(view2, index=index) 
        sim = self.sim(kgv1_ro, kgv2_ro) 
        return sim


    def get_adj_mat(self, tmp_adj):
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum+1e-8, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(
            coo.shape)).coalesce().to(configs['device'])
        g.requires_grad = False
        return g

    def ui_batch_drop_weighted(self, item_mask, start, end):
        item_mask = item_mask.cpu().numpy()
        n_nodes = self.Kg_model.num_users + self.Kg_model.num_items 
        item_np = self.Kg_model.dataset.trainItem 
        user_np = self.Kg_model.dataset.trainUser 
        indices = np.where((user_np >= start) & (user_np < end))[0] 
        batch_item = item_np[indices]  
        batch_user = user_np[indices] 
        keep_idx = list() 
        for u, i in zip(batch_user, batch_item):
            if item_mask[u - start, i]: 
                keep_idx.append([u, i])

        keep_idx = np.array(keep_idx) 
        user_np = keep_idx[:, 0] 
        item_np = keep_idx[:, 1] + self.Kg_model.num_users 
        ratings = np.ones_like(user_np, dtype=np.float32) 
        tmp_adj = sp.csr_matrix(   
            (ratings, (user_np, item_np)),
            shape=(n_nodes, n_nodes)) 
        return tmp_adj

    def get_ui_views_weighted_with_uemb(self, item_stabilities, user_score, start, end, init_view):
        user_item_stabilities = F.softmax(user_score, dim=-1) * item_stabilities 
        k = (1 - 0.6) / (user_item_stabilities.max() - user_item_stabilities.min())
        weights = 0.6 + k * (user_item_stabilities - user_item_stabilities.min()) 
        item_mask = torch.bernoulli(weights).to(torch.bool) 
        tmp_adj = self.ui_batch_drop_weighted(item_mask, start, end) 
        if init_view != None:
            tmp_adj = init_view + tmp_adj
        return tmp_adj

    def get_ui_kg_view(self, aug_side="both"):
        if aug_side == "ui":
            kgv1, kgv2 = None, None
            kgv3, kgv4 = None, None
        else:
            kgv1, kgv2 = self.get_kg_views()
            kgv3, kgv4 = self.get_kg_views()

        stability1 = self.item_kg_stability(kgv1, kgv2, index=0).to(configs['device']) 
        stability2 = self.item_kg_stability(kgv3, kgv4, index=1).to(configs['device']) 
        u = self.Kg_model.embedding_user.weight 
        i1 = self.Kg_model.emb_item_list[0].weight 
        i2 = self.Kg_model.emb_item_list[1].weight 

        user_length = self.Kg_model.train_n_users #2174
        #print('user_length', user_length)
        size = 2048
        step = user_length // size # 2
        init_view1, init_view2 = None, None
        for s in range(step):
            start = s * size
            end = (s + 1) * size
            u_i_s1 = u[start:end] @ i1.T 
            u_i_s2 = u[start:end] @ i2.T
            uiv1_batch_view = self.get_ui_views_weighted_with_uemb(stability1, u_i_s1, start, end, init_view1)
            uiv2_batch_view = self.get_ui_views_weighted_with_uemb(stability2, u_i_s2, start, end, init_view2)
            init_view1 = uiv1_batch_view 
            init_view2 = uiv2_batch_view
        uiv1 = self.get_adj_mat(init_view1)
        uiv2 = self.get_adj_mat(init_view2)

        contrast_views = {
            "uiv1": uiv1,
            "uiv2": uiv2
        }
        return contrast_views

# from utility.batch_test import data_generator
class KGModel(nn.Module):
    def __init__(self, dataset, kg_dataset): #dataset,kg_raw dataset
        super(KGModel, self).__init__()
        self.dataset = dataset 
        self.kg_dataset = kg_dataset
        self.__init_weight()
        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()
        self.train_n_users = self.dataset.train_n_user

    def __init_weight(self):
        self.num_users = self.dataset.n_users 
        self.num_items = self.dataset.m_items
        # self.num_items = self.dataset.m_items + 15 #######################################################################problem: need optimize for retail datasets
        # self.num_items = data_generator.n_items
        self.num_entities = self.kg_dataset.entity_count 
        self.num_relations = self.kg_dataset.relation_count 
        print("user:{}, item:{}, entity:{}".format(self.num_users,
                                                   self.num_items,
                                                   self.num_entities))
        self.latent_dim = configs['model']['latent_dim_rec'] # 32
        self.n_layers = configs['model']['lightGCN_n_layers'] # 3
        self.keep_prob = configs['model']['keep_prob']
        self.A_split = configs['model']['A_split']

        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim) 
        self.emb_item_list = nn.ModuleList([
            torch.nn.Embedding(self.num_items, self.latent_dim), 
            torch.nn.Embedding(self.num_items, self.latent_dim)
        ])
        self.emb_entity_list = nn.ModuleList([
            nn.Embedding(self.num_entities + 1, self.latent_dim), 
            nn.Embedding(self.num_entities + 1, self.latent_dim)
        ])
        self.emb_relation_list = nn.ModuleList([
            nn.Embedding(self.num_relations + 1, self.latent_dim), 
            nn.Embedding(self.num_relations + 1, self.latent_dim)
        ])

        for i in range(2):
            nn.init.normal_(self.emb_item_list[i].weight, std=0.1) 
            nn.init.normal_(self.emb_entity_list[i].weight, std=0.1) 
            nn.init.normal_(self.emb_relation_list[i].weight, std=0.1) 

        self.transR_W = nn.Parameter(torch.Tensor(self.num_relations + 1, self.latent_dim, self.latent_dim)) 
        self.TATEC_W = nn.Parameter(torch.Tensor(self.num_relations + 1, self.latent_dim, self.latent_dim)) 

        nn.init.xavier_uniform_(self.transR_W, gain=nn.init.calculate_gain('relu')) 
        nn.init.xavier_uniform_(self.TATEC_W, gain=nn.init.calculate_gain('relu'))

        self.W_R = nn.Parameter(
            torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim)) 
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu')) 
        nn.init.normal_(self.embedding_user.weight, std=0.1) 

        self.co_user_score = nn.Linear(self.latent_dim, 1) 
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(
            self.num_items) 


    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)  


    def cal_item_embedding_from_kg(self, kg: dict = None, index=0):
        if kg is None:
            kg = self.kg_dict

        return self.cal_item_embedding_rgat(kg, index)


    def cal_item_embedding_rgat(self, kg: dict, index):
        item_embs = self.emb_item_list[index](          
            torch.IntTensor(list(kg.keys())).to(
                configs['device']))
        item_entities = torch.stack(list(               
            kg.values()))
        item_relations = torch.stack(list(self.item2relations.values())).to(
                configs['device'])
        entity_embs = self.emb_entity_list[index](      
            item_entities)
        relation_embs = self.emb_relation_list[index](  
            item_relations)
        padding_mask = torch.where(item_entities != self.num_entities,  
                                   torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs,   
                                         padding_mask)


    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer() 

        users_emb = all_users[users] 
        pos_emb = all_items[pos_items] 
        neg_emb = all_items[neg_items] 

        users_emb_ego = self.embedding_user(users) 

        pos_emb_ego0 = self.emb_item_list[0](pos_items)
        pos_emb_ego1 = self.emb_item_list[1](pos_items)
        neg_emb_ego0 = self.emb_item_list[0](neg_items)
        neg_emb_ego1 = self.emb_item_list[1](neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego0, pos_emb_ego1, neg_emb_ego0, neg_emb_ego1

    def getAll(self):
        all_users, all_items = self.computer()
        return all_users, all_items


    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, pos_emb_ego0,
         pos_emb_ego1, neg_emb_ego0, neg_emb_ego1) = self.getEmbedding(users.long(), pos.long(), neg.long()) 
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + pos_emb_ego0.norm(2).pow(2) + pos_emb_ego1.norm(2).pow(2)
                              + neg_emb_ego0.norm(2).pow(2) + neg_emb_ego1.norm(2).pow(2)) / float(len(users))  # device='cuda:0'
        pos_scores = torch.mul(users_emb, pos_emb) 
        pos_scores = torch.sum(pos_scores, dim=1) 
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        return loss, reg_loss

    def computer(self):
        users_emb = self.embedding_user.weight 

        items_emb0 = self.cal_item_embedding_from_kg(index=0)
        items_emb1 = self.cal_item_embedding_from_kg(index=1) 

        items_emb = (items_emb0 + items_emb1) / 2  

        all_emb = torch.cat([users_emb, items_emb]) 
        embs = [all_emb]
        if configs['model']['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers): 
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb) 
        embs = torch.stack(embs, dim=1) 
        light_out = torch.mean(embs, dim=1) 
        users, items = torch.split(light_out, [self.num_users, self.num_items]) 
        return users, items 

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob) 
        return graph

    def __dropout_x(self, x, keep_prob):
        size = x.size()  
        index = x.indices().t() 
        values = x.values() 
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g


    def view_computer_all(self, g_droped, users_emb, items_emb, index): # add target-embedding #torch.Size([21716, 128]),[7977,128]
        # users_emb = self.embedding_user.weight
        # items_emb = self.cal_item_embedding_from_kg(index=index)
        all_emb = torch.cat([users_emb, items_emb]) #torch.Size([21716+7977, 128])

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1) #torch.Size([29693, 4, 128])
        light_out = torch.mean(embs, dim=1) #torch.Size([29693, 128])
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def calc_kg_loss_transR(self, h, r, pos_t, neg_t, index): 
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1) 
        h_embed = self.emb_item_list[index](h).unsqueeze(-1) 
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1) 
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1)  

        r_matrix = self.transR_W[r] 
        h_embed = torch.matmul(r_matrix, h_embed) 
        pos_t_embed = torch.matmul(r_matrix, pos_t_embed) 
        neg_t_embed = torch.matmul(r_matrix, neg_t_embed) 

        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), 
                              dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2),  
                              dim=1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)    
        kg_loss = torch.mean(kg_loss)  
        l2_loss = self._L2_loss_mean(h_embed) + self._L2_loss_mean(   
            r_embed) + self._L2_loss_mean(pos_t_embed) + self._L2_loss_mean(neg_t_embed) + torch.norm(self.transR_W)  #tensor(device='cuda:0')

        loss = kg_loss + 1e-3 * l2_loss 

        return loss

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t, index): 
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1) 
        h_embed = self.emb_item_list[index](h).unsqueeze(-1) 
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1) 
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1)  

        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), 
                              dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2),  
                              dim=1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)    
        kg_loss = torch.mean(kg_loss)  
        l2_loss = self._L2_loss_mean(h_embed) + self._L2_loss_mean(   
            r_embed) + self._L2_loss_mean(pos_t_embed) + self._L2_loss_mean(neg_t_embed) #tensor(device='cuda:0')

        loss = kg_loss + 1e-3 * l2_loss 

        return loss



    def calc_kg_loss_TATEC(self, h, r, pos_t, neg_t, index):
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1) 
        h_embed = self.emb_item_list[index](h).unsqueeze(-1) 
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1) 
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1) 

        r_matrix = self.TATEC_W[r] 
        pos_mrt = torch.matmul(r_matrix, pos_t_embed) 
        neg_mrt = torch.matmul(r_matrix, neg_t_embed) 

        pos_hmrt = torch.sum(h_embed * pos_mrt, dim=1)  # vh(T)·Mr·vt 
        neg_hmrt = torch.sum(h_embed * neg_mrt, dim=1)

        hr = torch.sum(h_embed * r_embed, dim=1) # vh(T)·vr 
        pos_tr = torch.sum(pos_t_embed * r_embed, dim=1) # vt(T)·vr
        neg_tr = torch.sum(neg_t_embed * r_embed, dim=1)

        pos_ht = torch.sum(h_embed * pos_t_embed, dim=1) # vh(T)·vt
        neg_ht = torch.sum(h_embed * neg_t_embed, dim=1)

        pos_score = pos_hmrt + hr + pos_tr + pos_ht #
        neg_score = neg_hmrt + hr + neg_tr + neg_ht

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score) #
        kg_loss = torch.mean(kg_loss)
        l2_loss = self._L2_loss_mean(h_embed) + self._L2_loss_mean(
            r_embed) + self._L2_loss_mean(pos_t_embed) + self._L2_loss_mean(neg_t_embed) + torch.norm(self.TATEC_W)

        loss = kg_loss + 1e-3 * l2_loss

        return loss



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha): #GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2)
        super(GAT, self).__init__()
        self.dropout = dropout #0.4
        self.num_heads = 1
        self.layers = nn.ModuleList([GraphAttentionLayer(nfeat,nhid,dropout=dropout,alpha=alpha,concat=True) for _ in range(self.num_heads)])
        self.out = nn.Linear(nhid * self.num_heads, nhid) #（32，32）

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.out(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_relation(self, item_embs, entity_embs, w_r, adj): #(item_embs, entity_embs, relation_embs, padding_mask)
        x = F.dropout(entity_embs, self.dropout, training=self.training) #
        x = torch.cat([att.forward_relation(item_embs, x, w_r, adj) for att in self.layers ], dim=1) # 

        x = self.out(x + item_embs) # sum(Wire·ve)+vi-l
        x = F.relu(x) #
        x = F.dropout(x, self.dropout, training=self.training) # item embedding
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))) #
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1))) #
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.fc = nn.Linear(in_features * 3, 1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations, adj):
        Wh = item_embs.unsqueeze(1).expand(entity_embs.shape[0],entity_embs.shape[1], -1) #
        We = entity_embs #
        e_input = self.fc(torch.cat([Wh, relations, We], dim=-1)).squeeze() # 
        e = self.leakyrelu(e_input) 

        zero_vec = -9e15 * torch.ones_like(e) 
        attention = torch.where(adj > 0, e, zero_vec) 
        attention = F.softmax(attention, dim=1) 
        attention = F.dropout(attention, self.dropout,training=self.training)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted
        return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs,self.W)
        We = torch.matmul(entity_embs, self.W)
        a_input = self._prepare_cat(Wh, We)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) 

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,training=self.training)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1),entity_embs).squeeze()
        h_prime = entity_emb_weighted + item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size())
        return torch.cat((Wh, We), dim=-1)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class BPRLoss:
    def __init__(self, recmodel, opt):
        self.model = recmodel
        self.opt = opt
        self.weight_decay = configs["model"]["decay"]

    def compute(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay 
        return loss



class RecLoss(nn.Module):
    def __init__(self, data_config, args):
        super(RecLoss, self).__init__()
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size
        self.coefficient = eval(args.coefficient)
        self.wid = eval(args.wid)

    def forward(self, input_u, label_phs, ua_embeddings, ia_embeddings, rela_embeddings):
        uid = ua_embeddings[input_u]
        uid = torch.reshape(uid, (-1, self.n_relations, self.emb_dim))
        pos_r_list = []
        for i in range(self.n_relations):
            beh = self.behs[i]
            pos_beh = ia_embeddings[:, i, :][label_phs[i]]  # [B, max_item, dim]
            pos_num_beh = torch.ne(label_phs[i], self.n_items).float()
            pos_beh = torch.einsum('ab,abc->abc', pos_num_beh,
                                   pos_beh)  # [B, max_item] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ac,abc->abc', uid[:, i, :],
                                 pos_beh)  # [B, dim] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ajk,lk->aj', pos_r, rela_embeddings[beh])
            pos_r_list.append(pos_r)

        loss = 0.
        for i in range(self.n_relations):
            beh = self.behs[i]
            temp = torch.einsum('ab,ac->bc', ia_embeddings[:, i, :], ia_embeddings[:, i, :]) \
                   * torch.einsum('ab,ac->bc', uid[:, i, :], uid[:, i, :])  # [B, dim]' * [B, dim] -> [dim, dim]
            tmp_loss = self.wid[i] * torch.sum(
                temp * torch.matmul(rela_embeddings[beh].T, rela_embeddings[beh]))
            tmp_loss += torch.sum((1.0 - self.wid[i]) * torch.square(pos_r_list[i]) - 2.0 * pos_r_list[i])

            loss += self.coefficient[i] * tmp_loss

        regularizer = torch.sum(torch.square(uid)) * 0.5 + torch.sum(torch.square(ia_embeddings)) * 0.5
        emb_loss = args.decay * regularizer

        return loss, emb_loss


class SSLoss2(nn.Module):
    def __init__(self, data_config, args):
        super(SSLoss2, self).__init__()
        self.config = data_config
        self.ssl_temp = args.ssl_temp
        self.ssl_reg_inter = eval(args.ssl_reg_inter)
        self.ssl_mode_inter = args.ssl_inter_mode
        self.topk1_user = args.topk1_user
        self.topk1_item = args.topk1_item
        self.user_indices_remove, self.item_indices_remove = None, None

    def forward(self, input_u_list, input_i_list, ua_embeddings, ia_embeddings, aux_beh, user_batch_indices=None,
                item_batch_indices=None):
        ssl2_loss = 0.

        if self.ssl_mode_inter in ['user_side', 'both_side']:
            emb_tgt = ua_embeddings[input_u_list, -1, :]  # [B, d]
            normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
            emb_aux = ua_embeddings[input_u_list, aux_beh, :]  # [B, d]
            normalize_emb_aux = F.normalize(emb_aux, dim=1)  # [B, dim]
            normalize_all_emb_aux = F.normalize(ua_embeddings[:, aux_beh, :], dim=1)  # [N, dim]
            pos_score = torch.sum(torch.mul(normalize_emb_tgt, normalize_emb_aux),
                                  dim=1)  # [B, ]
            ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)  # [B, N]

            ttl_score[user_batch_indices] = 0.
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)

            ssl2_loss += -torch.sum(torch.log(pos_score / ttl_score)) * self.ssl_reg_inter[aux_beh]

        if self.ssl_mode_inter in ['item_side', 'both_side']:
            emb_tgt = ia_embeddings[input_i_list, -1, :]
            normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
            emb_aux = ia_embeddings[input_i_list, aux_beh, :]
            normalize_emb_aux = F.normalize(emb_aux, dim=1)  # [B, dim]
            normalize_all_emb_aux = F.normalize(ia_embeddings[:, aux_beh, :], dim=1)  # [N, dim]
            pos_score = torch.sum(torch.mul(normalize_emb_tgt, normalize_emb_aux),
                                  dim=1)
            ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)
            #ttl_score[item_batch_indices] = 0.
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl2_loss += -torch.sum(torch.log(pos_score / ttl_score)) * self.ssl_reg_inter[aux_beh]

        return ssl2_loss

