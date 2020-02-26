#!/usr/bin/env python3
#
import math, logging, copy, json
from collections import Counter, OrderedDict
from nltk.util import ngrams

import pdb

import os
import sys
import subprocess
import threading

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100



# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:
    def __init__(self,language):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', language]
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        self.meteor_p.stdin.flush()
        for i in range(0,len(imgIds)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().decode().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()

def maskedPerplexity(seq, gtSeq):
    '''
    Compute the Perplexity of ground truth (target) sentence given the
    model. Assumes that gtSeq has <START> and <END> token surrounding
    every sequence and gtSeq is left aligned (i.e. right padded)
    S: <START>, E: <END>, W: word token, 0: padding token, P(*): logProb
        gtSeq:
            [ S     W1    W2  E   0   0]
        Teacher forced logProbs (seq):
            [P(W1) P(W2) P(E) -   -   -]
        Required gtSeq (target):
            [  W1    W2    E  0   0   0]
        Mask (non-zero tokens in target):
            [  1     1     1  0   0   0]
    '''
    # Shifting gtSeq 1 token left to remove <START>
    padColumn = gtSeq.data.new(gtSeq.size(0), 1).fill_(0)
    padColumn = Variable(padColumn)
    target = torch.cat([gtSeq, padColumn], dim=1)[:, 1:]

    # Generate a mask of non-padding (non-zero) tokens
    mask = target.data.gt(0)
    perplexity = 0
    # print("gtSeq type: {}".format(gtSeq.type()))
    if isinstance(gtSeq, Variable):
        # mask = Variable(mask, volatile=gtSeq.volatile)
        mask = Variable(mask)
    assert isinstance(target, Variable)
    # print(seq.size())
    # print(target.size())
    gtLogProbs = torch.gather(seq, 2, target.unsqueeze(2)).squeeze(2)
    # Mean sentence probs:
    gtLogProbs = gtLogProbs / (mask.float().sum(1).view(-1, 1))
    # print(gtLogProbs)
    # if returnScores:
    #     return (gtLogProbs * (mask.float())).sum(1)
    # maskedLL = torch.masked_select(gtLogProbs, mask)
    # perplexity = -torch.sum(maskedLL) / seq.size(0)
    perplexity = -(gtLogProbs * (mask.float())).sum(1)
    # print(perplexity)
    # print(2**perplexity.data.item())
    # print(type(perplexity.data.item()))
    return torch.exp(perplexity)
    # return perplexity


def concatPaddedSequences(seq1, seqLens1, seq2, seqLens2, padding='right'):
    '''
    Concates two input sequences of shape (batchSize, seqLength). The
    corresponding lengths tensor is of shape (batchSize). Padding sense
    of input sequences needs to be specified as 'right' or 'left'
    Args:
        seq1, seqLens1 : First sequence tokens and length
        seq2, seqLens2 : Second sequence tokens and length
        padding        : Padding sense of input sequences - either
                         'right' or 'left'
    '''

    concat_list = []
    cat_seq = torch.cat([seq1, seq2], dim=1)
    maxLen1 = seq1.size(1)
    maxLen2 = seq2.size(1)
    maxCatLen = cat_seq.size(1)
    batchSize = seq1.size(0)
    for b_idx in range(batchSize):
        # len_1 = seqLens1[b_idx].data[0]
        # len_2 = seqLens2[b_idx].data[0]
        len_1 = seqLens1[b_idx].data.item()
        len_2 = seqLens2[b_idx].data.item()

        cat_len_ = len_1 + len_2
        if cat_len_ == 0:
            raise RuntimeError("Both input sequences are empty")

        elif padding == 'left':
            pad_len_1 = maxLen1 - len_1
            pad_len_2 = maxLen2 - len_2
            if len_1 == 0:
                print("[Warning] Empty input sequence 1 given to "
                      "concatPaddedSequences")
                cat_ = seq2[b_idx][pad_len_2:]

            elif len_2 == 0:
                print("[Warning] Empty input sequence 2 given to "
                      "concatPaddedSequences")
                cat_ = seq1[b_idx][pad_len_1:]

            else:
                cat_ = torch.cat([seq1[b_idx][pad_len_1:],
                                  seq2[b_idx][pad_len_2:]], 0)
            cat_padded = F.pad(
                input=cat_,  # Left pad
                pad=((maxCatLen - cat_len_), 0),
                mode="constant",
                value=0)
        elif padding == 'right':
            if len_1 == 0:
                print("[Warning] Empty input sequence 1 given to "
                      "concatPaddedSequences")
                cat_ = seq2[b_idx][:len_1]

            elif len_2 == 0:
                print("[Warning] Empty input sequence 2 given to "
                      "concatPaddedSequences")
                cat_ = seq1[b_idx][:len_1]

            else:
                cat_ = torch.cat([seq1[b_idx][:len_1],
                                  seq2[b_idx][:len_2]], 0)
                # cat_ = cat_seq[b_idx].masked_select(cat_seq[b_idx].ne(0))
            cat_padded = F.pad(
                input=cat_,  # Right pad
                pad=(0, (maxCatLen - cat_len_)),
                mode="constant",
                value=0)
        else:
            raise (ValueError, "Expected padding to be either 'left' or \
                                'right', got '%s' instead." % padding)
        concat_list.append(cat_padded.unsqueeze(0))
    concat_output = torch.cat(concat_list, 0)
    return concat_output

if __name__ == '__main__':
    pass

