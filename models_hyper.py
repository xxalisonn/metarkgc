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

    def forward(self, M, N):
        M = torch.squeeze(M)
        N = torch.squeeze(N)
        # calculate the similarity score(mask the attn value of ground truth relation)
        mt = torch.matmul(N, M.transpose(0, 1))
        diag = mt - torch.diag(torch.diag(mt))
        attn_weight = torch.softmax(diag, dim=-1)
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

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r,pos_num):					# revise
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score

class Classifier(nn.Module):
    def __init__(self,embed_dim):
        super(Classifier, self).__init__()
        self.class_matrix = nn.Linear(embed_dim,5)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self,rel):
        rel = torch.squeeze(rel)
        class_result = self.class_matrix(rel)
        tag_ = [i for i in range(5)]
        tag = torch.tensor(tag_, dtype=torch.long).cuda()
        loss = self.criterion(class_result,tag)
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
        self.embedding = Embedding(dataset, parameter)
        self.pattern_learner = PatternLearner(input_channels=1)
        self.attention_matcher = AttentionMatcher(self.embed_dim)
        self.relation_classifer = Classifier(self.embed_dim)
        self.relation_prototype = dict()
        self.relation_minus = dict()
        self.rel_q_sharing = dict()

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

    def get_relation(self, pairs, mean=False):
        """return (tail-head)"""
        relation = (pairs[:, :, :, 1, :] - pairs[:, :, :, 0, :]).unsqueeze(-2)
        if mean:
            relation = torch.mean(relation, dim=1, keepdim=True)
        return relation

    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        pos_relation = self.get_relation(support.unsqueeze(2), mean=True)
        support_concat = self.concat_relation(support.unsqueeze(2), pos_relation)

        # support.squeeze(2)
        # rel = self.relation_learner(support)

        rel = self.pattern_learner(support_concat).unsqueeze(2)
        rel = torch.mean(rel, dim=1, keepdim=True)

        rel = self.attention_matcher(rel, pos_relation.squeeze(2))
        class_loss = self.relation_classifer(rel)


        # elif iseval:
        #     rel_ = rel.data
        #     for i in range(4):
        #         key = random.choice(list(self.relation_prototype.keys()))
        #         _ = self.relation_prototype[key].unsqueeze(1)
        #         __ = self.relation_minus[key].unsqueeze(1)
        #         rel_ = torch.cat((rel_,_),dim = 0)
        #         pos_relation = torch.cat((pos_relation,__),dim = 0)
        #
        #     rel_att = self.attention_matcher(rel_, pos_relation.squeeze(2))
        #     class_loss = self.relation_classifer(rel_att, iseval)
        #     rel = rel_att[0].unsqueeze(1)

        # relation for support
        rel.retain_grad()
        rel_s = rel[0].unsqueeze(1).expand(-1, few+num_sn, -1, -1)

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s,few)

                y = torch.ones(p_score.size()).cuda()
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y) + class_loss
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta

                if not iseval:
                    idx = 0
                    for relation in curr_rel:
                        self.relation_prototype[relation] = rel_q[idx]
                        self.relation_minus[relation] = pos_relation[idx]
                        idx += 1
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)

        return p_score, n_score

