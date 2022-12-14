from embedding import *
from hyper_embedding import *
from collections import OrderedDict
import torch.nn.functional as F
import torch
from torch.nn.modules.conv import Conv1d, Conv2d
from torch.nn.modules.activation import ReLU
import random

class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)

class AttentionMatcher(nn.Module):
    def __init__(self,embed_dim):
        super(AttentionMatcher, self).__init__()
        # attn_mat = nn.Parameter(torch.randn(embed_dim))
        self.gate_w = nn.Linear(embed_dim,1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

    def forward(self, M, N,iseval):
        M = torch.squeeze(M)
        if not iseval:
            N = torch.squeeze(N)
        # calculate the similarity score(mask the attn value of ground truth relation)
        mt = torch.matmul(N, M.transpose(0, 1))
        if not iseval:
            diag = mt - torch.diag(torch.diag(mt))
            attn_weight = torch.softmax(diag, dim=-1)
        elif iseval:
            mt[0] = 0
            attn_weight = torch.softmax(mt, dim=-1)
        # multiply the score with relation prototype
        out_attn = torch.matmul(attn_weight, M)
        # set a gate for fusion(0<=gate<=1)
        gate_tmp = self.gate_w(out_attn) + self.gate_b
        gate = torch.sigmoid(gate_tmp)
        # gate*boosted + (1-gate)*original
        out_rel = torch.mul(out_attn,gate)
        boosted = out_rel + torch.mul(N,1.0-gate)

        return boosted.unsqueeze(1).unsqueeze(1)

class PatternLearner(nn.Module):
    def __init__(self, input_channels, out_channels=[128, 64, 32, 1]):
        super(PatternLearner, self).__init__()
        self.out_channels = out_channels
        self.input_channles = input_channels
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Conv2d(self.input_channles, self.out_channels[0], (1, 3)),
                    ),
                    # BatchNorm2d(128),
                    ("relu1", ReLU()),
                    (
                        "conv2",
                        Conv2d(self.out_channels[0], self.out_channels[1], (1, 1)),
                    ),
                    # BatchNorm2d(64),
                    ("relu2", ReLU()),
                    (
                        "conv3",
                        Conv2d(self.out_channels[1], self.out_channels[2], (1, 1)),
                    ),
                    # BatchNorm2d(32),
                    ("relu3", ReLU()),
                    (
                        "conv4",
                        Conv2d(self.out_channels[2], self.out_channels[3], (1, 1)),
                    ),
                ]
            )
        )

        for name, param in self.encoder.named_parameters():
            if "conv" and "weight" in name:
                torch.nn.init.kaiming_normal_(param)

    def forward(self, x):
        batch_size, num_triples, num_channels, input_length, dim = x.size()
        x = x.view(batch_size * num_triples, num_channels, input_length, dim).transpose(
            2, 3
        )
        x = self.encoder(x)
        return x.view(batch_size, num_triples, -1)

# class EmbeddingLearner(nn.Module):
#     def __init__(self):
#         super(EmbeddingLearner, self).__init__()
#
#     def forward(self, h, t, r,pos_num):					# revise
#         score = -torch.norm(h + r - t, 2, -1).squeeze(2)
#         p_score = score[:, :pos_num]
#         n_score = score[:, pos_num:]
#         return p_score, n_score

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, hp, tp, rp, pos_num, alpha):					# revise
        score_minus = -torch.norm(h + r - t, 2, -1).squeeze(2)
        score_plus = -torch.norm(hp - rp - tp, 2, -1).squeeze(2)
        score = alpha * score_plus + score_minus
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score

class Classifier(nn.Module):
    def __init__(self,embed_dim,num_rel):
        super(Classifier, self).__init__()
        self.class_matrix = nn.Linear(embed_dim,num_rel)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self,rel,tag,iseval):
        # if iseval:
        #     print('eval',rel.size())
        # elif not iseval:
        #     print('print',rel.size())
        rel = torch.squeeze(rel)
        if iseval:
            rel = rel.unsqueeze(0)
        class_result = self.class_matrix(rel)
        tag_ = torch.tensor(tag, dtype=torch.long).cuda()
        loss = self.criterion(class_result,tag_)
        return loss

class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.alpha = parameter['alpha']
        self.bs = parameter['batch_size']
        self.rel2id = dataset['rel2id']
        self.all_num = len(self.rel2id)
        self.plus_embedding = Embedding(dataset, parameter)
        self.embedding = Embedding(dataset, parameter)
        self.pattern_learner = PatternLearner(input_channels=1)
        self.attention_matcher = AttentionMatcher(self.embed_dim)
        self.relation_classifer = Classifier(self.embed_dim,self.all_num)
        self.relation_prototype = dict()
        self.relation_minus = dict()
        self.rel_q_sharing = dict()
        self.rel_q_plus_sharing = dict()

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)

        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def concat_relation(self, pairs, relation):
        if relation.shape[1] != pairs.shape[1]:
            relation = relation.repeat(1, pairs.shape[1], 1, 1, 1)
        triplet = torch.cat((relation, pairs), dim=-2)
        return triplet[:, :, :, [1, 0, 2], :]

    def get_relation(self, pairs, pattern,mean=False):
        """return (tail-head)"""
        if pattern == 'minus':
            relation = (pairs[:, :, :, 1, :] - pairs[:, :, :, 0, :]).unsqueeze(-2)
        if pattern == 'add':
            relation = (pairs[:, :, :, 1, :] + pairs[:, :, :, 0, :]).unsqueeze(-2)
        if pattern == 'concat':
            relation = torch.concat((pairs[:, :, :, 1, :] + pairs[:, :, :, 0, :]),dim=3).unsqueeze(-2)
        if mean:
            relation = torch.mean(relation, dim=1, keepdim=True)
        return relation

    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        support_plus, support_negative_plus, query_plus, negative_plus = [self.plus_embedding(t) for t in task]
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        pos_relation_minus = self.get_relation(support.unsqueeze(2),'minus',mean=True)
        pos_relation_plus = self.get_relation(support_plus.unsqueeze(2),'add',mean=True)
        support_concat = self.concat_relation(support.unsqueeze(2), pos_relation_minus)
        support_concat_plus = self.concat_relation(support_plus.unsqueeze(2), pos_relation_plus)

        rel = self.pattern_learner(support_concat).unsqueeze(2)
        rel_plus = self.pattern_learner(support_concat_plus).unsqueeze(2)

        rel = torch.mean(rel, dim=1, keepdim=True)
        rel_plus = torch.mean(rel, dim=1, keepdim=True)



        # curr_rel_id = []
        # if not iseval:
        #     idx = 0
        #     for relation in curr_rel:
        #         curr_rel_id.append(self.rel2id[relation])
        #         self.relation_prototype[relation] = rel[idx]
        #         # self.relation_minus[relation] = pos_relation[idx]
        #         idx += 1
        #     rel = self.attention_matcher(rel, pos_relation_minus.squeeze(2), iseval)
        #     class_loss = self.relation_classifer(rel, curr_rel_id, iseval)
        #
        # elif iseval:
        #     # rel_boosted = rel.clone()
        #     curr_rel_id.append(self.rel2id[curr_rel])
        #     keylist = list(self.relation_prototype.keys())
        #     rel_boosted = self.relation_prototype[keylist[0]].unsqueeze(2)
        #     for _ in self.relation_prototype.keys():
        #         rel_boosted = torch.cat((rel_boosted,self.relation_prototype[_].unsqueeze(2)),0)
        #     rel_ = rel_boosted.clone().detach()
        #     rel = self.attention_matcher(rel_, pos_relation_minus.squeeze(2), iseval).squeeze(2).squeeze(2)
        #     class_loss = self.relation_classifer(rel, curr_rel_id,iseval)

        # support.squeeze(2)
        # rel = self.relation_learner(support)
        # relation for support

        # because in test and dev step, same relation uses same support, so it's no need to repeat the step of relation-meta learning
        rel.retain_grad()
        rel_s = rel.expand(-1, few + num_sn, -1, -1)
        rel_plus.retain_grad()
        rel_s_plus = rel_plus.expand(-1, few + num_sn, -1, -1)

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
            rel_q_plus = self.rel_q_plus_sharing[curr_rel]
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
                sup_neg_e1_plus, sup_neg_e2_plus = self.split_concat(support_plus, support_negative_plus)

                # p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s,few)
                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, sup_neg_e1_plus, sup_neg_e2_plus, rel_plus, few, self.alpha)

                y = torch.ones(p_score.size()).cuda()
                self.zero_grad()
                # loss = self.loss_func(p_score, n_score, y) + class_loss
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                grad_meta_plus = rel_plus.grad
                rel_q = rel - self.beta*grad_meta
                rel_q_plus = rel_plus - self.beta*grad_meta_plus

            else:
                rel_q = rel
                rel_q_plus = rel_plus

            self.rel_q_sharing[curr_rel] = rel_q
            self.rel_q_plus_sharing[curr_rel] = rel_q_plus

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        rel_q_plus = rel_q_plus.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        que_neg_e1_plus, que_neg_e2_plus = self.split_concat(query_plus, negative_plus)
        # p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, que_neg_e1_plus, que_neg_e2_plus, rel_q_plus, num_q,self.alpha)

        return p_score, n_score
