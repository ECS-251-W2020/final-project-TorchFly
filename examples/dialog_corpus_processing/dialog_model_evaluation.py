import sys
import json
import h5py
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import options
import visdial.metrics as metrics
from utils import utilities as utils
from dataloader import VisDialDataset
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import pairwise_distances

from six.moves import range
import visdial.loss.loss_utils as loss_utils
from visdial.loss.infoGain import txtPrior, prepareBatch, normProb
from visdial.loss.rl_txtuess_loss import Ranker, rl_rollout_search

pairwiseRanking_criterion = loss_utils.PairwiseRankingLoss(margin=0.1)


def txtLoader(dataloader, dataset):
    all_txt_feat = []
    for idx, batch in enumerate(dataloader):
        batch = prepareBatch(dataset, batch)
        all_txt_feat.append(Variable(batch['txt_feat'], requires_grad=False))
    all_txt_feat = torch.cat(all_txt_feat, 0)
    return all_txt_feat


def DialogEval(val_model, dataset, split, exampleLimit=None, verbose=0, txt_retrieval_mode='mse'):
    print("text retrieval mode is: {}".format(txt_retrieval_mode))
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    # enumerate all gt features and all predicted features
    gttxtFeatures = []
    # caption + dialog rounds
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]
    logProbsAll = [[] for _ in range(numRounds)]
    featLossAll = [[] for _ in range(numRounds + 1)]
    # Added by Mingyang Zhou for Perplexity Computation
    perplexityAll = [[] for _ in range(numRounds)]
    start_t = timer()

    # Modified by Mingyang Zhou
    # Record the wining rates for the questioner in multiple games
    win_rate = [0] * (numRounds + 1)
    num_games = 0

    # Modified by Mingyang Zhou
    all_txt_feat = txtLoader(dataloader, dataset)
    im_ranker = Ranker()

    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        else:
            batch = {
                key: v.contiguous()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        # caption = Variable(batch['cap'], volatile=True)
        # captionLens = Variable(batch['cap_len'], volatile=True)
        # gtQuestions = Variable(batch['ques'], volatile=True)
        # gtQuesLens = Variable(batch['ques_len'], volatile=True)
        # answers = Variable(batch['ans'], volatile=True)
        # ansLens = Variable(batch['ans_len'], volatile=True)
        # gtFeatures = Variable(batch['txt_feat'], volatile=True)
        with torch.no_grad():
            caption = Variable(batch['cap'])
            captionLens = Variable(batch['cap_len'])
            gtQuestions = Variable(batch['ques'])
            gtQuesLens = Variable(batch['ques_len'])
            answers = Variable(batch['ans'])
            ansLens = Variable(batch['ans_len'])
            if txt_retrieval_mode == "mse":
                if val_model.txtEncodingMode == "txtuess":
                    gtFeatures = val_model.forwardtext(Variable(batch['txt_feat']))
                else:
                    gtFeatures = Variable(batch['txt_feat'])
            else:
                gtFeatures = Variable(batch['txt_feat'])
                gtFeatures = val_model.multimodalpredictIm(gtFeatures)
            text = Variable(batch['txt_feat'])  # Added by Mingyang Zhou
            # Update the Ranker
            if val_model.txtEncodingMode == "txtuess":
                im_ranker.update_rep(val_model, all_txt_feat)

            val_model.reset()
            val_model.observe(-1, caption=caption, captionLens=captionLens)
            if val_model.new_questioner:
                val_model.observe_txt(text)

            if val_model.txtEncodingMode == "txtuess":
                act_index = torch.randint(
                    0, all_txt_feat.size(0) - 1, (text.size(0), 1))
                predicted_text = all_txt_feat[act_index].squeeze(1)
                val_model.observe_txt(predicted_text)

            if txt_retrieval_mode == "mse":
                predFeatures = val_model.predicttext()
                # Evaluating round 0 feature regression network
                featLoss = F.mse_loss(predFeatures, gtFeatures)
                #featLoss = F.mse_loss(predFeatures, gtFeatures)
                featLossAll[0].append(torch.mean(featLoss))
                # Keeping round 0 predictions
                roundwiseFeaturePreds[0].append(predFeatures)

                # Modified by Mingyang Zhou for txtEncoding Mode == "txtuess"
                if val_model.txtEncodingMode == "txtuess":
                    # act_index = im_ranker.nearest_neighbor(
                    #     predFeatures.data, all_txt_feat)
                    act_index = im_ranker.nearest_neighbor(
                        predFeatures.data)
                    predicted_text = all_txt_feat[act_index]

                # Compute the winning rate at round 0, modified by Mingyang
                # Zhou
                round_dists = pairwise_distances(
                    predFeatures.cpu().numpy(), gtFeatures.cpu().numpy())

                for i in range(round_dists.shape[0]):
                    current_rank = int(
                        np.where(round_dists[i, :].argsort() == i)[0]) + 1
                    if current_rank <= 1:
                        win_rate[0] += 1
                    # update the num_games
                    num_games += 1

            elif txt_retrieval_mode == "cosine_similarity":
                dialogEmbedding = val_model.multimodalpredictText()
                featLoss = pairwiseRanking_criterion(
                    gtFeatures, dialogEmbedding)
                featLossAll[0].append(torch.sum(featLoss))
                roundwiseFeaturePreds[0].append(
                    dialogEmbedding)
                # Initailize the round_dists, with each row as the cosine
                # similarity
                round_dists = np.matmul(
                    dialogEmbedding.cpu().numpy(), gtFeatures.cpu().numpy().transpose())
                for i in range(round_dists.shape[0]):
                    current_rank = int(
                        np.where(round_dists[i, :].argsort()[::-1] == i)[0]) + 1
                    if current_rank <= 1:
                        win_rate[0] += 1
                    # update the num_games
                    num_games += 1

            # convert gtFeatures back to tensor
            # gtFeatures = torch.from_numpy(gtFeatures)

            for round in range(numRounds):
                if val_model.txtEncodingMode == "txtuess":
                    val_model.observe_txt(predicted_text)
                val_model.observe(
                    round,
                    ques=gtQuestions[:, round],
                    quesLens=gtQuesLens[:, round])
                val_model.observe(
                    round, ans=answers[:, round], ansLens=ansLens[:, round])
                logProbsCurrent = val_model.forward()

                # Evaluating logProbs for cross entropy
                logProbsAll[round].append(
                    utils.maskedNll(logProbsCurrent,
                                    gtQuestions[:, round].contiguous()))
                perplexityAll[round].append(utils.maskedPerplexity(logProbsCurrent,
                                                                   gtQuestions[:, round].contiguous()))

                if txt_retrieval_mode == "mse":
                    predFeatures = val_model.predicttext()
                    # Evaluating feature regression network

                    # Deal with different txtEncodingMode
                    featLoss = F.mse_loss(predFeatures, gtFeatures)

                    featLossAll[round + 1].append(torch.mean(featLoss))
                    # Keeping predictions
                    roundwiseFeaturePreds[round + 1].append(predFeatures)

                    # Modified by Mingyang Zhou
                    if val_model.txtEncodingMode == "txtuess":
                        # act_index = im_ranker.nearest_neighbor(
                        #     predFeatures.data, all_txt_feat)
                        act_index = im_ranker.nearest_neighbor(
                            predFeatures.data)
                        predicted_text = all_txt_feat[act_index].squeeze(1)

                    # Compute the winning rate at round 0, modified by Mingyang
                    # Zhou
                    round_dists = pairwise_distances(
                        predFeatures.cpu().numpy(), gtFeatures.cpu().numpy())
                    for i in range(round_dists.shape[0]):
                        current_rank = int(
                            np.where(round_dists[i, :].argsort() == i)[0]) + 1
                        if current_rank <= 1:
                            win_rate[round + 1] += 1

                elif txt_retrieval_mode == "cosine_similarity":
                    dialogEmbedding = val_model.multimodalpredictText()
                    featLoss = pairwiseRanking_criterion(
                        gtFeatures, dialogEmbedding)
                    featLossAll[round + 1].append(torch.sum(featLoss))
                    roundwiseFeaturePreds[round + 1].append(
                        dialogEmbedding)  # Keep the dialogEmbedding, To be modified later.
                    # Initailize the round_dists, with each row as the cosine
                    # similarity
                    round_dists = np.matmul(
                        dialogEmbedding.cpu().numpy(), gtFeatures.cpu().numpy().transpose())
                    for i in range(round_dists.shape[0]):
                        current_rank = int(
                            np.where(round_dists[i, :].argsort()[::-1] == i)[0]) + 1
                        if current_rank <= 1:
                            win_rate[round + 1] += 1

                # convert gtFeatures back to tensor
                # gtFeatures = torch.from_numpy(gtFeatures)

            gttxtFeatures.append(gtFeatures)

            end_t = timer()
            delta_t = " Time: %5.2fs" % (end_t - start_t)
            start_t = end_t
            progressString = "\r[val_model] Evaluating split '%s' [%d/%d]\t" + delta_t
            sys.stdout.write(progressString % (split, idx + 1, numBatches))
            sys.stdout.flush()

    sys.stdout.write("\n")
    # Compute the win_rate, modified by Mingyang Zhou
    win_rate = [x / num_games for x in win_rate]
    print("The winning rates for {} are: {}".format(split, win_rate))

    gtFeatures = torch.cat(gttxtFeatures, 0).data.cpu().numpy()
    rankMetricsRounds = []
    poolSize = len(dataset)

    # Keeping tracking of feature regression loss and CE logprobs
    # logProbsAll = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll]
    # featLossAll = [torch.cat(floss, 0).mean() for floss in featLossAll]
    # roundwiseLogProbs = torch.cat(logProbsAll, 0).data.cpu().numpy()
    # roundwiseFeatLoss = torch.cat(featLossAll, 0).data.cpu().numpy()
    logProbsAll = [torch.stack(lprobs, 0).mean() for lprobs in logProbsAll]
    # Compute the Mean Perplexity for each round
    perplexityAll = [torch.cat(perplexity, 0).mean().data.item()
                     for perplexity in perplexityAll]

    featLossAll = [torch.stack(floss, 0).mean() for floss in featLossAll]
    roundwiseLogProbs = torch.stack(logProbsAll, 0).data.cpu().numpy()
    roundwiseFeatLoss = torch.stack(featLossAll, 0).data.cpu().numpy()
    # Compute the Mean Perplexity over all rounds
    # roundwisePerplexity = torch.stack(perplexityAll, 0).data.cpu().numpy()
    logProbsMean = roundwiseLogProbs.mean()
    featLossMean = roundwiseFeatLoss.mean()
    perplexityMean = sum(perplexityAll) / len(perplexityAll)
    print("The Perplxity of current Questioner is: {}".format(perplexityMean))
    # Added by Mingyang Zhou
    winrateMean = sum(win_rate) / len(win_rate)

    if verbose:
        print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):
        if txt_retrieval_mode == "mse":
            predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                     0).data.cpu().numpy()
            # num_examples x num_examples
            dists = pairwise_distances(predFeatures, gtFeatures)
            ranks = []
            for i in range(dists.shape[0]):
                rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
                ranks.append(rank)
        elif txt_retrieval_mode == "cosine_similarity":
            predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                     0).data.cpu().numpy()
            dists = np.matmul(predFeatures, gtFeatures.transpose())
            ranks = []
            for i in range(dists.shape[0]):
                rank = int(np.where(dists[i, :].argsort()[::-1] == i)[0]) + 1
                ranks.append(rank)

        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        if verbose:
            print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank
        rankMetrics['featLoss'] = roundwiseFeatLoss[round]
        if round < len(roundwiseLogProbs):
            rankMetrics['logProbs'] = roundwiseLogProbs[round]
        rankMetricsRounds.append(rankMetrics)

    rankMetricsRounds[-1]['logProbsMean'] = logProbsMean
    rankMetricsRounds[-1]['featLossMean'] = featLossMean
    rankMetricsRounds[-1]['winrateMean'] = winrateMean
    # Added the perplexity in eval metrics
    rankMetricsRounds[-1]['perplexityMean'] = perplexityMean

    dataset.split = original_split
    return rankMetricsRounds[-1], rankMetricsRounds


def DialogEval_2(val_model, target_model, dataset, split, exampleLimit=None, beamSize=1, txt_retrieval_mode='mse'):
    print("text Encoding Mode is: {}".format(val_model.txtEncodingMode))
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    gttxtFeatures = []
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]
    # Added by Mingyang Zhou for Perplexity Computation
    # perplexityAll = [[] for _ in range(numRounds)]

    start_t = timer()

    # Defined by Mingyang Zhou
    win_rate = [0] * (numRounds + 1)
    num_games = 0

    # Modified by Mingyang Zhou
    all_txt_feat = txtLoader(dataloader, dataset)
    im_ranker = Ranker()

    # Update the Ranker
    val_model.eval(), val_model.reset()
    if val_model.txtEncodingMode == "txtuess":
        im_ranker.update_rep(val_model, all_txt_feat)

    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {key: v.cuda() for key, v in batch.items()
                     if hasattr(v, 'cuda')}
        else:
            batch = {key: v.contiguous() for key, v in batch.items()
                     if hasattr(v, 'cuda')}

        # caption = Variable(batch['cap'], volatile=True)
        # captionLens = Variable(batch['cap_len'], volatile=True)
        # gtQuestions = Variable(batch['ques'], volatile=True)
        # gtQuesLens = Variable(batch['ques_len'], volatile=True)
        # answers = Variable(batch['ans'], volatile=True)
        # ansLens = Variable(batch['ans_len'], volatile=True)
        # gtFeatures = Variable(batch['txt_feat'], volatile=True)
        # text = Variable(batch['txt_feat'], volatile=True)
        with torch.no_grad():
            caption = Variable(batch['cap'])
            captionLens = Variable(batch['cap_len'])
            gtQuestions = Variable(batch['ques'])
            gtQuesLens = Variable(batch['ques_len'])
            answers = Variable(batch['ans'])
            ansLens = Variable(batch['ans_len'])
            if txt_retrieval_mode == "mse":
                if val_model.txtEncodingMode == "txtuess":
                    gtFeatures = val_model.forwardtext(Variable(batch['txt_feat']))
                else:
                    gtFeatures = Variable(batch['txt_feat'])
            else:
                gtFeatures = Variable(batch['txt_feat'])
                gtFeatures = val_model.multimodalpredictIm(gtFeatures)
            text = Variable(batch['txt_feat'])

            target_model.eval(), target_model.reset()
            target_model.observe(-1, text=text, caption=caption,
                         captionLens=captionLens)
            val_model.eval(), val_model.reset()
            val_model.observe(-1, caption=caption, captionLens=captionLens)
            if val_model.new_questioner:
                val_model.observe_txt(text)

            if val_model.txtEncodingMode == "txtuess":
                act_index = torch.randint(
                    0, all_txt_feat.size(0) - 1, (text.size(0), 1))
                predicted_text = all_txt_feat[act_index].squeeze(1)
                val_model.observe_txt(predicted_text)

            if txt_retrieval_mode == "mse":
                predFeatures = val_model.predicttext()
                roundwiseFeaturePreds[0].append(predFeatures)

                # Modified by Mingyang Zhou for txtEncoding Mode == "txtuess"
                if val_model.txtEncodingMode == "txtuess":
                    # act_index = im_ranker.nearest_neighbor(
                    #     predFeatures.data, all_txt_feat)
                    act_index = im_ranker.nearest_neighbor(
                        predFeatures.data)
                    predicted_text = all_txt_feat[act_index]
                    # Should observe the current predicted text
                    val_model.observe_txt(predicted_text)

                # Compute the winning rate at round 0, modified by Mingyang
                # Zhou
                round_dists = pairwise_distances(
                    predFeatures.cpu().numpy(), gtFeatures.cpu().numpy())
                for i in range(round_dists.shape[0]):
                    current_rank = int(
                        np.where(round_dists[i, :].argsort() == i)[0]) + 1
                    if current_rank <= 1:
                        win_rate[0] += 1
                    # update the num_games
                    num_games += 1
            elif txt_retrieval_mode == "cosine_similarity":
                dialogEmbedding = val_model.multimodalpredictText()
                roundwiseFeaturePreds[0].append(
                    dialogEmbedding)
                # Initailize the round_dists, with each row as the cosine
                # similarity
                round_dists = np.matmul(
                    dialogEmbedding.cpu().numpy(), gtFeatures.cpu().numpy().transpose())
                for i in range(round_dists.shape[0]):
                    current_rank = int(
                        np.where(round_dists[i, :].argsort()[::-1] == i)[0]) + 1
                    if current_rank <= 1:
                        win_rate[0] += 1
                    # update the num_games
                    num_games += 1

            for round in range(numRounds):
                # questions, quesLens = val_model.forwardDecode(
                #     inference='greedy', beamSize=beamSize)
                questions, quesLens = val_model.forwardDecode(
                    inference='greedy', beamSize=beamSize)
                # print(logProbsCurrent.size())
                val_model.observe(round, ques=questions, quesLens=quesLens)
                target_model.observe(round, ques=questions, quesLens=quesLens)
                # answers, ansLens = target_model.forwardDecode(
                #     inference='greedy', beamSize=beamSize)
                answers, ansLens = target_model.forwardDecode(
                    inference='greedy', beamSize=beamSize)
                target_model.observe(round, ans=answers, ansLens=ansLens)
                val_model.observe(round, ans=answers, ansLens=ansLens)
                if val_model.new_questioner:
                    val_model.observe_txt(text)
                if val_model.txtEncodingMode == "txtuess":
                    val_model.observe_txt(predicted_text)

                # Added by Mingyang Zhou
                # logProbsCurrent = val_model.forward()
                # perplexityAll[round].append(utils.maskedPerplexity(logProbsCurrent,
                # gtQuestions[:, round].contiguous()))
                if txt_retrieval_mode == "mse":
                    predFeatures = val_model.predicttext()
                    roundwiseFeaturePreds[round + 1].append(predFeatures)

                    # Modified by Mingyang Zhou for txtEncoding Mode ==
                    # "txtuess"
                    if val_model.txtEncodingMode == "txtuess":
                        # act_index = im_ranker.nearest_neighbor(
                        #     predFeatures.data, all_txt_feat)
                        act_index = im_ranker.nearest_neighbor(
                            predFeatures.data)
                        predicted_text = all_txt_feat[act_index]
                    # Compute the winning rate at round 0, modified by Mingyang
                    # Zhou
                    round_dists = pairwise_distances(
                        predFeatures.cpu().numpy(), gtFeatures.cpu().numpy())
                    for i in range(round_dists.shape[0]):
                        current_rank = int(
                            np.where(round_dists[i, :].argsort() == i)[0]) + 1
                        if current_rank <= 1:
                            win_rate[round + 1] += 1
                elif txt_retrieval_mode == "cosine_similarity":
                    dialogEmbedding = val_model.multimodalpredictText()
                    roundwiseFeaturePreds[round + 1].append(
                        dialogEmbedding)  # Keep the dialogEmbedding, To be modified later.
                    # Initailize the round_dists, with each row as the cosine
                    # similarity
                    round_dists = np.matmul(
                        dialogEmbedding.cpu().numpy(), gtFeatures.cpu().numpy().transpose())
                    for i in range(round_dists.shape[0]):
                        current_rank = int(
                            np.where(round_dists[i, :].argsort()[::-1] == i)[0]) + 1
                        if current_rank <= 1:
                            win_rate[round + 1] += 1

            gttxtFeatures.append(gtFeatures)

            end_t = timer()
            delta_t = " Rate: %5.2fs" % (end_t - start_t)
            start_t = end_t
            progressString = "\r[val_model] Evaluating split '%s' [%d/%d]\t" + delta_t
            sys.stdout.write(progressString % (split, idx + 1, numBatches))
            sys.stdout.flush()
    sys.stdout.write("\n")
    # Compute the win_rate, modified by Mingyang Zhou
    win_rate = [x / num_games for x in win_rate]
    print("The winning rates for {} are: {}".format(split, win_rate))

    gtFeatures = torch.cat(gttxtFeatures, 0).data.cpu().numpy()
    rankMetricsRounds = []
    # Added by Mingyang Zhou
    # perplexityAll = [sum(perplexity) / len(perplexity)
    #                  for perplexity in perplexityAll]
    # perplexityMean = sum(perplexityAll) / len(perplexityAll)
    # print("The Perplxity of current Questioner in the Dialog with a User Simulator is: {}".format(
    #     perplexityMean))

    winrateMean = sum(win_rate) / len(win_rate)
    print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):
        if txt_retrieval_mode == "mse":
            predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                     0).data.cpu().numpy()
            dists = pairwise_distances(predFeatures, gtFeatures)
            # num_examples x num_examples
            ranks = []
            for i in range(dists.shape[0]):
                # Computing rank of i-th prediction vs all texts in split
                rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
                ranks.append(rank)
        elif txt_retrieval_mode == "cosine_similarity":
            predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                     0).data.cpu().numpy()
            dists = np.matmul(predFeatures, gtFeatures.transpose())
            ranks = []
            for i in range(dists.shape[0]):
                rank = int(np.where(dists[i, :].argsort()[::-1] == i)[0]) + 1
                ranks.append(rank)

        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        assert len(ranks) == len(dataset)
        poolSize = len(dataset)
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank
        rankMetricsRounds.append(rankMetrics)

    dataset.split = original_split
    rankMetricsRounds[-1]['winrateMean'] = winrateMean
    return rankMetricsRounds[-1], rankMetricsRounds



