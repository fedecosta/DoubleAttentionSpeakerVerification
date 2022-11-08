import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import math
import copy

def new_parameter(*size):
    #print("[poolings] Using new_parameter...")
    out = torch.nn.Parameter(torch.FloatTensor(*size))
    #print(f"[poolings] out size : {out.size()}")
    torch.nn.init.xavier_normal_(out)
    #print(f"[poolings] out size after xn init: {out.size()}")
    return out


class Attention(nn.Module):

    def __init__(self, embedding_size):

        super().__init__()
        self.embedding_size = embedding_size
        self.att = new_parameter(self.embedding_size, 1)

    def forward(self, ht):
        #print(f"[poolings] ht size : {ht.size()}")
        attention_score = torch.matmul(ht, self.att)
        #print(f"[poolings] attention_score size : {attention_score.size()}")
        attention_score = attention_score.squeeze()
        #print(f"[poolings] attention_score size : {attention_score.size()}")
        attention_score = F.softmax(attention_score, dim = -1)
        #print(f"[poolings] attention_score size : {attention_score.size()}")
        attention_score = attention_score.view(ht.size(0), ht.size(1), 1)
        #print(f"[poolings] attention_score size : {attention_score.size()}")
        ct = torch.sum(ht * attention_score, dim = 1)
        #print(f"[poolings] ct size : {ct.size()}")

        #print(f"[poolings] Attention output size : {ct.size()}")
        
        return ct, attention_score


class HeadAttention(nn.Module):

    def __init__(self, encoder_size, heads_number, mask_prob = 0.25, attentionSmoothing=False):

        super(HeadAttention, self).__init__()
        self.embedding_size = encoder_size//heads_number
        self.att=new_parameter(self.embedding_size,1)
        self.mask_prob = int(1/mask_prob)
        self.attentionSmoothing = attentionSmoothing

    def __maskAttention(self, attention_score, mask_value = -float('inf')):
        
        # seems that this works only with GPU (what about CPU?)
        mask = torch.cuda.FloatTensor(attention_score.size()).random_(self.mask_prob)>0
        attention_score[~mask] = mask_value
        return attention_score

    def __narrowAttention(self, new_ht):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score)
        attention_score = F.softmax(attention_score, dim=-1).view(new_ht.size(0), new_ht.size(1),1)
        return attention_score 

    def __wideAttention(self):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score, mask_value = -1)
        attention_score /= torch.sum(attention_score, dim=1).unsqueeze(1)
        return attention_score.view(new_ht.size(0), new_ht.size(1),1)

    def forward(self,ht):

        if self.attentionSmoothing:
            attention_score = self.__wideAttention(ht)
        else:
            attention_score = self.__narrowAttention(ht)

        weighted_ht = ht * attention_score
        ct = torch.sum(weighted_ht,dim=1)

        # print(f"[poolings] HeadAttention output size : {ct.size()}")

        return ct, attention_score


def innerKeyValueAttention(query, key, value):

    d_k = query.size(-1)
    scores = torch.diagonal(torch.matmul(key, query) / math.sqrt(d_k), dim1=-2, dim2=-1).view(value.size(0),value.size(1), value.size(2))
    p_attn = F.softmax(scores, dim = -2)
    weighted_vector = value * p_attn.unsqueeze(-1)
    ct = torch.sum(weighted_vector, dim=1)
    return ct, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, encoder_size, heads_number):
        super(MultiHeadAttention, self).__init__()
        self.encoder_size = encoder_size
        assert self.encoder_size % heads_number == 0 # d_model
        self.head_size = self.encoder_size // heads_number 
        self.heads_number = heads_number
        self.query = new_parameter(self.head_size, self.heads_number)
        self.aligmment = None
        
    def getAlignments(self,ht): 
        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return self.alignment 
    
    def getHeadsContextVectors(self,ht):    
        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return headsContextVectors

    def forward(self, ht):
        headsContextVectors = self.getHeadsContextVectors(ht)

        #print(f"[poolings] MultiHeadAttention output size : {headsContextVectors.view(headsContextVectors.size(0),-1).size()}")

        return headsContextVectors.view(headsContextVectors.size(0),-1), copy.copy(self.alignment)


class DoubleMHA(nn.Module):
    def __init__(self, encoder_size, heads_number, mask_prob=0.2):
        super(DoubleMHA, self).__init__()
        self.heads_number = heads_number
        self.utteranceAttention = MultiHeadAttention(encoder_size, heads_number)
        self.heads_size = encoder_size // heads_number
        self.headsAttention = HeadAttention(encoder_size, heads_number, mask_prob=mask_prob, attentionSmoothing=False)

    def getAlignments(self, x):

        utteranceRepresentation, alignment = self.utteranceAttention(x)
        headAlignments = self.headsAttention(utteranceRepresentation.view(utteranceRepresentation.size(0), self.heads_number, self.heads_size))[1]
        return alignment, headAlignments

    def forward(self, x):
        utteranceRepresentation, alignment = self.utteranceAttention(x)
        compressedRepresentation = self.headsAttention(utteranceRepresentation.view(utteranceRepresentation.size(0), self.heads_number, self.heads_size))[0]    
        
        # print(f"[poolings] DoubleMHA output size : {compressedRepresentation.size()}")
        
        return compressedRepresentation, alignment


# ----------------------------------------------------------------------------
# new classes

# TODO make dim asserts in every new class

class SelfAttention(nn.Module):
  
  def __init__(self):
    
    super().__init__()

  def forward(self, x):

    raw_weights = torch.bmm(x, x.transpose(1, 2))

    weights = F.softmax(raw_weights, dim = 2)

    y = torch.bmm(weights, x)

    return y


class MultiHeadAttention2(nn.Module):

    def __init__(self, emb_in, emb_out, heads=8):
        """
        :param emb_in: dimension of the embeddings input vectors
        :param emb_in: dimension of the embeddings output vectors
        :param heads: number of heads to use
        :param mask: ?
        """

        super().__init__()

        self.emb_in = emb_in
        self.emb_out = emb_out
        self.heads = heads

        self.to_keys = nn.Linear(emb_in, emb_out * heads, bias=False)
        self.to_queries = nn.Linear(emb_in, emb_out * heads, bias=False)
        self.to_values = nn.Linear(emb_in, emb_out * heads, bias=False)

        self.unify_heads = nn.Linear(heads * self.emb_out, self.emb_out)

    def forward(self, x):

        b, t, e = x.size()
        assert e == self.emb_in, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb_in})'

        keys = self.to_keys(x).view(b, t, self.heads, self.emb_out)
        queries = self.to_queries(x).view(b, t, self.heads, self.emb_out)
        values = self.to_values(x).view(b, t, self.heads, self.emb_out)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, self.emb_out)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(self.emb_out) # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b * self.heads, t, t), f'Matrix has size {dot.size()}, expected {(b * self.heads, t, t)}.'

        dot = F.softmax(dot, dim = 2) # dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, self.heads, t, self.emb_out)

        # swap h, t back
        out = out.transpose(1, 2).contiguous().view(b, t, self.heads * self.emb_out)

        # unify heads
        out = self.unify_heads(out)

        return out


class StatisticalPooling(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, hidden_states):

        # Get the average of the hidden states (dim = 0 is the batch dimension)
        context_vector = hidden_states.mean(dim = 1)
        
        # Returning a tuple because the other pooling methods do this
        return context_vector, None


class AttentionPooling(nn.Module):

    def __init__(self, emb_in):

        super().__init__()

        self.emb_in = emb_in

        self.query = torch.nn.Parameter(torch.FloatTensor(self.emb_in, 1))
        torch.nn.init.xavier_normal_(self.query)

    def forward(self, x):

        attention_scores = torch.matmul(x, self.query)
        attention_scores = attention_scores.squeeze(dim = -1)
        attention_scores = F.softmax(attention_scores, dim = 1)
        attention_scores = attention_scores.unsqueeze(dim = -1)

        output_embedding = torch.bmm(attention_scores.transpose(1, 2), x)

        return output_embedding, attention_scores


class SelfAttentionAttentionPooling(nn.Module):

    def __init__(self, emb_in):

        super().__init__()

        self.emb_in = emb_in
        self.self_attention = SelfAttention()
        self.attention_pooling = AttentionPooling(emb_in)

    def forward(self, x):

        output = self.self_attention(x)
        output_embedding, attention_scores = self.attention_pooling(output)

        return output_embedding, attention_scores


class MultiHeadAttentionAttentionPooling(nn.Module):

    def __init__(self, emb_in, emb_out, heads):

        super().__init__()

        self.emb_in, self.emb_out, self.heads = emb_in, emb_out, heads
        self.mha = MultiHeadAttention2(self.emb_in, self.emb_out, self.heads)
        self.attention_pooling = AttentionPooling(self.emb_out)

    def forward(self, x):

        output = self.mha(x)
        output_embedding, attention_scores = self.attention_pooling(output)

        return output_embedding, attention_scores
        



    

















