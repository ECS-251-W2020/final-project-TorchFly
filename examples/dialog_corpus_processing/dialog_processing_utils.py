import praw
import redis
import string
from get_news import RNews_Client
import signal
import time
import re
from nltk import sent_tokenize
import requests
import json
import random
import logging
import os
import boto3
from boto3.dynamodb.conditions import Key, Attr
from pprint import pprint
from twitterES import moment_search
#from service_client import get_client

# For Mining Opinions
from textblob import TextBlob
from difflib import SequenceMatcher
from dialog_utterance_data import dialog_utterance
import template_manager
from service_client import get_client

def detect_question_DA(dialog_act):
    r"""
    dialog_act is the dialog_act from Dian, which is a dictionary that contains the text for the sentence segment and the associated dialog act tag
    """
    if (dialog_act["DA"] == "yes_no_question" or dialog_act["DA"] == "open_question_factual" or dialog_act["DA"] == "open_question_opinion" or dialog_act["DA"] == "open_question") and float(dialog_act['confidence']) > 0.5:
        return (dialog_act['text'], True, float(dialog_act['confidence']))
    # elif amz_dialog_act == "Information_RequestIntent" or amz_dialog_act == "Opinion_RequestIntent":
    #     return (dialog_act['text'], True)
    else:
        return (None, False, 0)


def detect_question_DA_update(returnnlp_util, segment_index):
    if returnnlp_util.has_dialog_act([
        DialogActEnum.OPEN_QUESTION,
        DialogActEnum.OPEN_QUESTION_FACTUAL,
        DialogActEnum.OPEN_QUESTION_OPINION,
        DialogActEnum.YES_NO_QUESTION
    ], index=segment_index):
        return True
    else:
        return False


def detect_abandoned_answer_DA(dialog_act, sys_info):
    r"""
    Decode whether users have abandoned dialog_act
    """
    if dialog_act["DA"] == "abandoned" and float(dialog_act["confidence"]) > 0.9 and not sys_info['sys_noun_phrase']:
        return True
    return False


def detect_abandoned_answer_DA_update(returnnlp_util, segment_index, sys_info):
    """
    detect whether  users have abandoned  dialog_act
    """
    if returnnlp_util.has_dialog_act(DialogActEnum.ABANDONED) and not sys_info['sys_noun_phrase']:
        return True
    return False

# dialog Chat


def detect_dialog_general_chitchat_qa(asked_questions, selected_chitchat_key):
    talked_questions = access_nested_dict(
        asked_questions, selected_chitchat_key)
    if talked_questions:
        if len(talked_questions) < len(dialog_utterance(selected_chitchat_key, get_all=True)):
            return True
    else:
        return True
    return False


def detect_dialog_type(utterance):
    for dialog in MAIN_DIALOG_LIST:
        dialog_re = r"\b{dialog}\b".format(dialog=dialog)
        if re.search(dialog_re, utterance.lower()):
            return dialog
    return None


def detect_dialog_label(utterance):
    for label in ['player', 'team']:
        if re.search(label, utterance.lower()):
            return label
    return None

def detect_command(input_utterance, input_returnnlp):
    """
    Detect whether a command (not a device command) is given from the user
    """
    command_detector = CommandDetector(input_utterance, input_returnnlp)
   
    return command_detector.has_command()

def detect_same_yes_or_no(user_dialog_intent, system_response):
    answer_yes_or_no = False
    answer_same = False

    if user_dialog_intent['dialog_i_answeryes'] or user_dialog_intent['dialog_i_answerno']:
        answer_yes_or_no = True
        if user_dialog_intent['dialog_i_answeryes'] and re.search('yes', system_response.lower()):
            answer_same = True
        elif user_dialog_intent['dialog_i_answerno'] and re.search('no', system_response.lower()):
            answer_same = True

    return (answer_yes_or_no, answer_same)


def detect_ignored_dialog(utterance, ignore_pattern):
    r"""
    return a list of detected ignored dialog type in utterance
    """
    result = []
    for pattern in ignore_pattern:
        searched_word = re.search(pattern, utterance)
        if searched_word:
            result.append(searched_word.group())
    return result


# Create a function to filter out certain
def filter_entity_terms(entity):
    r"""
        return True if the entity should be filtered
    """
    result = False
    if entity in LEAGUE_DIALOG_MAP.keys():
        result = True
    return result


def stop_words_word2vec_disambiguater(utterance):
    r"""
    Return True if we should transit stop_words to word2vec
    """
    for word in word2vec_TERMS:
        if re.search(word, utterance.lower()):
            return True
    return False


def get_sys_dialog_entity(utterance, sys_info):
    r'''
    Return the best Quality Name Entity
    '''
    sys_ner = []
    sys_topic_ner = []
    sys_knowledge_ner = []
    knowledge_dialog_detected = False

    # get shte noun_phrases
    sys_noun_phrase = sys_info['sys_noun_phrase']

    if sys_info['sys_ner']:
        sys_ner = [x['text'].lower() for x in sys_info['sys_ner'] if x[
            'text'].lower() not in MAIN_dialog_LIST and x[
            'text'].lower() not in WRONG_NER]

    if sys_info['sys_topic']:
        if sys_info['sys_topic'][0].get('topicClass', None) == "dialogs":
            sys_topic_ner = [x['keyword'].lower()
                             for x in sys_info['sys_topic'][0]['topicKeywords'] if x[
                'keyword'].lower() not in MAIN_dialog_LIST]

    if sys_info['sys_knowledge']:
        for knowledge in sys_info['sys_knowledge']:
            # The second condition is to prevent the misclassification of dialog
            # type into entity.
            if knowledge:
                # logging.info(
                #     '[dialogS] i really hate this knowledge thing that continuously causes bugs:{}'.format(knowledge))
                if knowledge[3] == 'dialogs' and (float(knowledge[2]) > 50):
                    # and (knowledge[0].lower() in sys_noun_phrase)
                    for word in sys_noun_phrase:
                        word = word.lower()
                        if (word not in MAIN_dialog_LIST) and (word not in EXCEPTION_GOOGLE_KNOWLEDGE_KEY) and re.search(word, knowledge[0].lower()):
                            sys_knowledge_ner.append(knowledge)
                            knowledge_dialog_detected = True
                            break

    # Just for checking purpose
    # if sys_knowledge_ner:
    #     logging.info("[dialog MODULE] the detected google knowledge ner is : {}".format(
    #         sys_knowledge_ner[0]))

    result_ners = []
    if sys_topic_ner or sys_knowledge_ner:
        if sys_knowledge_ner and sys_topic_ner:
            for skn in sys_knowledge_ner:
                for stn in sys_topic_ner:
                    if re.search(stn, skn[0].lower()) or re.search(skn[0].lower(), stn):
                        selected_ner = {}
                        selected_ner['name'] = skn[0] if len(skn[0]) > len(
                            stn) else stn  # keep the longer one
                        selected_ner['label'] = detect_dialog_label(skn[1])
                        selected_ner['dialog_type'] = detect_dialog_type(skn[1])

                        if selected_ner['dialog_type'] == 'stop_words' and stop_words_word2vec_disambiguater(utterance):
                            selected_ner['dialog_type'] = 'word2vec'

                        if not filter_entity_terms(selected_ner['name']):
                            result_ners.append(selected_ner.copy())

        if sys_knowledge_ner and sys_ner:
            for skn in sys_knowledge_ner:
                for sn in sys_ner:
                    # Merge the result of entity, and amazon ner together
                    if re.search(sn, skn[0].lower()) or re.search(skn[0].lower(), sn):
                        selected_ner = {}
                        selected_ner['name'] = skn[0] if len(skn[0]) > len(
                            sn) else sn  # keep the longer one
                        selected_ner['label'] = detect_dialog_label(skn[1])
                        selected_ner['dialog_type'] = detect_dialog_type(skn[1])

                        if selected_ner['dialog_type'] == 'stop_words' and stop_words_word2vec_disambiguater(utterance):
                            selected_ner['dialog_type'] = 'word2vec'

                        if not filter_entity_terms(selected_ner['name']):
                            result_ners.append(selected_ner.copy())

        if sys_topic_ner and sys_ner:
            for stn in sys_topic_ner:
                for sn in sys_ner:
                    if re.search(sn, stn):
                        selected_ner = {}
                        selected_ner['name'] = stn
                        selected_ner['label'] = None
                        selected_ner['dialog_type'] = None
                        if not filter_entity_terms(selected_ner['name']):
                            result_ners.append(selected_ner.copy())
        if sys_knowledge_ner:
            for skn in sys_knowledge_ner:
                selected_ner = {}
                selected_ner['name'] = skn[0]
                selected_ner['label'] = detect_dialog_label(skn[1])
                selected_ner['dialog_type'] = detect_dialog_type(skn[1])

                if selected_ner['dialog_type'] == 'stop_words' and stop_words_word2vec_disambiguater(utterance):
                    selected_ner['dialog_type'] = 'word2vec'

                if not filter_entity_terms(selected_ner['name']):
                    result_ners.append(selected_ner.copy())
        '''
        elif sys_ner and 'topic_dialog' in sys_info['sys_intent']['topic']:
            selected_ner = {}
            selected_ner['name'] = sys_ner[0]
            selected_ner['label'] = None
            selected_ner['dialog_type'] = None
        '''
    #logging.info("[dialogS] All the detected ners: {}".format(result_ners))
    return result_ners


def ground_question_with_dialog_type(utterance, current_dialog, detected_dialogs):
    r"""
    Ground questions that has keyword player, game and team with dialog type if no dialog type is detected
    """
    result = utterance


    word_list = utterance.split()

    if not detected_dialogs and current_dialog:
        for key_word in grounding_keyword_list:
            if key_word in word_list:
                # find key_word index
                key_word_index = word_list.index(key_word)
                # insert the dialog type before index
                word_list.insert(key_word_index, current_dialog)

                return " ".join(word_list)
    return result


def detect_dialog_googlekg(sys_info):
    result = False
    for x in sys_info['sys_knowledge']:
        if x:
            if x[3] == 'dialogs':
                return True
    return False


def get_sys_other_topic(sys_info, detected_dialogs, utterance):
    r"""
    Return the other topic detected by system while no dialog intent is detected
    """
    detect_dialog = False
    all_sys_detected_topic = []

    for y in sys_info['sys_intent_2']:
        all_sys_detected_topic += y['topic']

    if 'topic_dialog' in all_sys_detected_topic or detect_dialog_googlekg(sys_info) or detected_dialogs or "dialog" in sys_info["sys_central_elem"]["module"]:
        detect_dialog = True

    if not detect_dialog:
        # check if the sys_intent list contains other topics
        for topic in sys_info['sys_intent']['topic']:
            if topic in SYS_OTHER_TOPIC_LIST:
                # Check if one of the topic we can talk is detected here
                if TOPIC_KEYWORD_MAP.get(topic, None):
                    return TOPIC_KEYWORD_MAP[topic]
                else:
                    return True

        # check if the sys_topic topic module contains other topics
        for module in sys_info['sys_central_elem']['module']:
            if module in SYS_OTHER_MODULE_LIST:
                if not any(re.search(x, utterance) for x in WRONG_JUMP_OUT.get(module, [])):
                    return MODULE_KEYWORD_MAP.get(module, True)
                #logging.debug("[dialogS_MODULE] wrong jumpout: {}".format(WRONG_JUMP_OUT.get(module, [])))

        # if sys_info['sys_module'] in SYS_OTHER_MODULE_LIST:
        #     return MODULE_KEYWORD_MAP.get(sys_info['sys_module'], True)

        # If topic is not detected

        # for knowledge in sys_info['sys_knowledge']:
        #     if knowledge:
        #         if knowledge[3] != 'dialogs' and knowledge[3] != '__EMPTY__' and float(knowledge[2]) > 500 and (knowledge[1].lower() not in EXCEPTION_GOOGLE_KNOWLEDGE_KEY):
        #             return True
    return None


def check_utterance_coherence(sys_info, detected_entities, user_context):
    r"""
    Check if the keywords responded from users is shared with the previous utterance
    """
    previous_response = user_context['previous_system_response']

    # A bit data processing
    if previous_response:
        re.sub(r'_', " ", previous_response)

        if detected_entities['sys_ner']:
            for ner in detected_entities['sys_ner']:
                #logging.info("[dialogS_MODULE] detected_entities: {}".format(ner))
                if re.search(ner['name'], previous_response.lower()):
                    return True
        elif sys_info['sys_noun_phrase']:
            for np in sys_info['sys_noun_phrase']:
                if re.search(np, previous_response.lower()):
                    return True

    return False


def check_talked_entities(test_entity, talked_entities):
    r"""
    Return True, if none of the repeated entities are found in talked entities. Vice Versa, Return False.
    """
    result = True
    for x in talked_entities:
        if re.search(test_entity, x.lower()):
            return False
    return result
# dialog Response F
# Detect dialog Type in utterance

# Define a function to create a dictionary_item


def create_qa_dict_from_list(qa_list, data):
    r'''
        Create a nested dictionary item from a list, where the list is formatted as:
        [key_1,key_2,..., key_3, value]
        qa_list: List of keys that you want to include in the nested dictionary
        data: the data you want to store in the last key of the dictionary
        return:
            a nested dictionary, with data as its ending data
    '''
    result_dict = {}
    for i, key in enumerate(reversed(qa_list)):
        if i == 0:
            result_dict[key] = data
        else:
            data = result_dict.copy()
            result_dict = {}
            result_dict[key] = data

    return result_dict


def access_nested_dict(nested_dict, key):
    r'''
    Give a list of keys for a nested data, return the data. If the key does not exist return None
    '''
    try:
        data = nested_dict[key[0]]
        for k in key[1:]:
            data = data[k]
        return data
    except Exception as e:
        return None


def update_nesteddict_with_new_key(nested_dict, key, value):
    r'''
    determine which part of the key is missing from the nested_dict
    '''
    data = nested_dict.get(key[0], None)
    if data:
        update_nesteddict_with_new_key(nested_dict[key[0]], key[1:], value)
    else:
        nested_dict[key[0]] = create_qa_dict_from_list(key, value)[key[0]]


def update_nesteddict_with_existing_key(nested_dict, key, updated_value):
    r'''
        Update the existing subset key at the right spot. The updated value should be updated at the end of the keys
    '''
    data = nested_dict[key[0]]
    if isinstance(data, dict):
        update_nesteddict_with_existing_key(
            nested_dict[key[0]], key[1:], updated_value)
    else:
        nested_dict[key[0]].append(updated_value)

#############################################Ground dialog Team Function###


def load_dialog_teams(dialog_type):
    r"""
    Load the list of teams belonging to a certain dialog type
    """
    result = None
    # Define the path to the dialog team data
    dialog_team_path = 'teams'
    # Check whether dialog_type has corresponding files
    team_file_name = '/'.join([dialog_team_path, "{}.txt".format(dialog_type)])
    if os.path.isfile(team_file_name):
        with open(team_file_name, 'r') as f:
            team_names = f.readlines()
        # clean the data and store it in the result
        result = [x.strip().lower() for x in team_names]

    return result


def load_all_dialog_teams():
    r"""
    Load all the existing teams and create a big list of them
    """
    dialog_team_path = 'teams'

    result = []
    for file in os.listdir(dialog_team_path):
        if file.endswith(".txt"):
            with open('/'.join([dialog_team_path, file]), 'r') as f:
                current_teams = f.readlines()
            result += [x.strip().lower() for x in current_teams]
    return result


#############################################API Functions################
def retrieve_reddit(utterance, subreddits, limit):
    # cuurently using gt's client info
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent='extractor', \
                         # username='gunrock_cobot', # not necessary for read-only
                         # password='ucdavis123'
                         )
    subreddits = '+'.join(subreddits)
    subreddit = reddit.subreddit(subreddits)
    # higher limit is slower; delete limit arg for top 100 results
    submissions = list(subreddit.search(utterance, limit=limit))
    if len(submissions) > 0:
        return submissions
    return None


def get_interesting_reddit_post_with_timeout(text, limit):
    try:
        with Timeout(2):
            reddit_post = get_interesting_reddit_posts(text, limit)

    except Timeout.Timeout:
        logging.info("get interesting reddit post timeout")
        return None

    return reddit_post


def get_interesting_reddit_posts(text, limit):
    submissions = retrieve_reddit(text, ['todayilearned'], limit)
    if submissions != None:
        results = [submissions[i].title.replace(
            "TIL", "").lstrip() for i in range(len(submissions))]
        results = [results[i] + '.' if results[i][-1] != '.' and results[i][-1] !=
                   '!' and results[i][-1] != '?' else results[i] for i in range(len(results))]
        # Sensitive Words Pattern
        final_results = []
        for result in results:
            # found_profanity = False
            # for w in SENSITIVE_WORDS_LIST:
            #     try:
            #         regex_w = re.compile(w)
            #         if re.search(regex_w, result.lower()):
            #             logging.debug(
            #                 "[dialogS_REDDIT]profanity word found is: {}".format(w))
            #             found_profanity = True
            #             break
            #     except:
            #         logging.debug(
            #             "[dialogS_REDDIT]issues with the word: {}".format(w))
            # if not found_profanity:
            #     final_results.append(result)
            found_profanity = BotResponseProfanityClassifier.is_profane(
                result)
            if not found_profanity:
                final_results.append(result)

        return final_results
    return []

# Define the Client for NewsBot
# Get the News Client


def get_popular_dialog_topics():
    news_bot = RNews_Client(addr='52.87.136.90')
    pop_themes = news_bot.get_pop_theme_with_timeout(
        'dialogs', top=5, timeout=2)
    return pop_themes


# results = get_interesting_reddit_posts('sandwich', limit=2)
# print(results)


def get_KG_news(entity, keyword=[], category=[]):
    r"""
Extract News from KG provided from Kevin
Input:
    entity: A list of entity names,
    keyword: A list of keywords to limit the search
"""
    news = kgapi_events(entity, keyword, category)
    if news:
        return news
    else:
        return None


def post_process_KG_news(content):
    content = content.lower()
    A = re.findall(r"\.{3,}", content)
    B = re.findall(r"[(].*[)]", content)
    if A:
        content = content.replace(A[0], '.')
    if B:
        content = content.replace(B[0], '')
    return content


def get_retrieval_response(utterance, sub=None):
    keys = redis_get_key.brpoplpush(RETRIEVAL_LIST_KEY, RETRIEVAL_LIST_KEY)
    c_id, c_secret, u_agent = keys.split("::")
    r = praw.Reddit(client_id=c_id, client_secret=c_secret, user_agent=u_agent)

    try:
        if sub == None:
            submission = r.subreddit('all').search(utterance, limit=n_titles)
        else:
            submission = r.subreddit(sub).search(utterance, limit=n_titles)

        t = 0
        weight = 1.5
        weight_decay = 0.6
        comments = []
        for s in submission:
            s.comment_limit = n_comments
            s.comment_sort = "top"
            t += 1
            if t > n_titles:
                break
            ups = int(s.ups)
            up_ratio = float(s.upvote_ratio)
            real_ups = ups

            if real_ups == 0:
                real_ups = 1000  # for scaling purpose

            if ups > 100:
                ups = 1.21
            elif ups > 30:
                ups = 1.1
            elif ups > 10:
                ups = 1
            else:
                ups = 0.9
            title = s.title
            # title = title + " " + s.selftext
            # TODO: add s.selftext, which is the detailed content of the
            # post(title)
            if profanity_check(title):
                continue
            c = 0
            for top_level_comment in s.comments:
                if c >= n_comments:
                    break
                top1_comment = top_level_comment.body
                if top_level_comment.author != None:
                    if top_level_comment.author.name == "AutoModerator":
                        break
                # print(top1_comment)
                # if profanity_check(top1_comment):
                #     continue
                top1_comment = filter_profanity(top1_comment)
                score = int(top_level_comment.score)

                # sentiment analysis, base = 2, give higher score to positive title+comment
                # title_sentiment_score = sid.polarity_scores(title)
                # comment_sentiment_score = sid.polarity_scores(top1_comment)
                # # more weights on the comments
                # totle_sentiment_score = title_sentiment_score["pos"] - title_sentiment_score["neg"] + 2 * (comment_sentiment_score["pos"] - comment_sentiment_score["neg"])
                # sentiment_weight = float(2 + totle_sentiment_score)
                sentiment_weight = 1
                comments.append((sentiment_weight * weight * (ups *
                                                              up_ratio * score / real_ups), (title, top1_comment)))
                c += 1
                weight *= weight_decay  # give more weights to the most relevant post

        comments = sorted(comments, key=lambda x: x[0], reverse=True)
        # print(comments)
        if len(comments) > 0:
            title1 = process_retrieve(comments[0][1][0], True)
            if title1 != "well i have a lot of thoughts on this, but it's not gonna be easy to talk about it now unless you are free until tomorrow night. maybe we should talk about something else":
                comment1 = process_retrieve(comments[0][1][1])
            #     template_idx = random.randint(0, len(title_template)-1)
            #     #removed the title that zhou doesn't like and the comment has to go since it cannot be mainted to be grammically correct or
            #     #of relevant value. ie the response 'its so obvious'
            #     response = title1
                response = title1 + '. ' + comment1
            else:
                response = title1
        else:
            response = None

    except:
        response = None

    return response


def filter_profanity(text):
    sentences = text.split(".")
    filtered_text = ""
    for sent in sentences:
        good = True
        for w in blacklist:
            if re.search(r"\b%s\b" % w, sent.lower()):
                good = False
                break
        if good:
            filtered_text = filtered_text + sent + "."
    return filtered_text


def profanity_check(text):
    # profanity check built by the client
    # r = client.batch_detect_profanity(utterances=[text])
    # the "1" here means the source is the statistical_model where if it is 0, it would be blacklist
    # return r["offensivenessClasses"][0]["values"][1]["offensivenessClass"]
    # return 1 means it is offensive; 0 means not offensive

    for w in blacklist:
        if re.search(r"\b%s\b" % w, text.lower()):
            return 1
    return 0


def process_retrieve(text, title=False):
    text = re.sub(r"https\S+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = text.replace("\n", ". ").strip()

    limit = title_len_limit if title else sent_len_limit

    num_words = len(text.split())
    spec_char = '#&\()*+-/:;<=>@[\\]^_`{|}~'
    if num_words <= 1:
        # fixed because zhou hates the previous template.
        return ". What do you think? I haven't made up my mind yet."
    if num_words <= sent_len_limit:
        text = re.sub('[%s]' % re.escape(spec_char), '', text)
        return text
    else:
        # print(text)
        sentences = re.split('\?|\.|\,|\!|\n', text)
        sentences = [re.sub('[%s]' % re.escape(spec_char), '', s)
                     for s in sentences]
        word_in_sent = 0
        for i in range(len(sentences)):
            word_in_sent += len(sentences[i].split())
            if word_in_sent > sent_len_limit:
                if i == 0:
                    return ". There are too many different opinions on this for me to decide. " \
                           "But I'm curious, can you tell me yours? "
                else:
                    return ". ".join(word for word in sentences[:i + 1])

# Functions to Extract the dialog Moment


def get_dialog_moment_data():
    type_tag = 'dialogs'
    response = table.query(
        # ProjectionExpression= "id,name,score,sub,title,opinion,related",
        IndexName='type',
        KeyConditionExpression=Key('type').eq(type_tag),
        ScanIndexForward=False
    )
    resp = response['Items']
    # print(type(resp))
    return resp


def get_moment_source_lookup(source):
    response = table.query(
        ProjectionExpression="title, #desc, #source, #type, opinions",
        ExpressionAttributeNames={"#desc": "desc",
                                  "#source": "source",
                                  "#type": "type", },
        IndexName='source',
        KeyConditionExpression=Key('source').eq(source),
        ScanIndexForward=False
    )
    resp = response['Items']

    result = []
    for item in resp:
        if item['type'] == 'dialogs':
            result.append(item)

    return result


def get_moment_title_lookup(np, limit=2):
    SCANLIMIT = 40
    start = time.time()
    if len(np) > 1:
        #fe = Attr('title').contains(np[0].lower()) & Attr('title').contains(np[1].lower())
        fe = Attr('title').contains(np[0].lower())
        for word in np[1:]:
            fe = fe & Attr('title').contains(word.lower())

    elif np:

        fe = Attr('title').contains(np[0].lower())
    else:
        return []

    response = table.scan(
        FilterExpression=fe,
    )

    resp = response['Items']
    i = 1
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response[
                              'LastEvaluatedKey'], FilterExpression=fe)
        resp.extend(response['Items'])
        stop = time.time()
        elapse = stop - start
        if len(resp) > 5 or elapse > 1:
            break
        i += 1
    stop = time.time()

    # Filter the resp by specifying the type
    for r in resp:
        if r['type'] != 'dialogs':
            resp.remove(r)
    '''
    if len(resp) > limit:
        resp = resp[:limit]
    '''

    return resp


def get_moment_desc_lookup(np, limit=2):
    SCANLIMIT = 40
    start = time.time()
    if len(np) > 1:
        #fe = Attr('title').contains(np[0].lower()) & Attr('title').contains(np[1].lower())
        fe = Attr('desc').contains(np[0].lower())
        for word in np[1:]:
            fe = fe & Attr('desc').contains(word.lower())

    elif np:

        fe = Attr('desc').contains(np[0].lower())
    else:
        return []

    response = table.scan(
        FilterExpression=fe,
    )

    resp = response['Items']
    i = 1
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response[
                              'LastEvaluatedKey'], FilterExpression=fe)
        resp.extend(response['Items'])
        stop = time.time()
        elapse = stop - start
        if len(resp) > 5 or elapse > 1:
            break
        i += 1
    # Filter the resp by specifying the type
    for r in resp:
        if r['type'] != 'dialogs':
            resp.remove(r)

    stop = time.time()
    '''
    if len(resp) > limit:
        resp = resp[:limit]
    '''
    return resp


def get_moment_keywords():
    v = REDIS_R.hget(KEYWORD_PREFIX, 'keys')
    return json.loads(v)


def format_moment_contents(moment_responses):
    processed_moments = []
    for m in moment_responses:
        m_title = m['title']  # str(m['title'],'utf-8')
        m_description = m['desc']  # str(m['desc'],'utf-8')
        m_result = '. '.join([m_title, m_description])
        processed_moments.append(m_result)
    processed_moments = list(set(processed_moments))
    return processed_moments


def search_moment_by_keywords(keyword, limit=2):
    try:
        # Search Moment with Keywords
        moments = get_moment_title_lookup([keyword])
        if not moments:
            moments = get_moment_desc_lookup([keyword])
        # # Back Up Option
        # if not moments:
        #     moments = moment_search(keyword)
        processed_moments = format_moment_contents(moments)
        if len(processed_moments) > limit:
            processed_moments = processed_moments[:limit]
    except:
        return []
    return processed_moments


def search_moment_by_dialog_type(dialog_type, limit=2):
    try:
        corresponding_moment_tags = dialog_MOMENT_TAG_MAP.get(dialog_type, [])
        moments = []
        # print(corresponding_moment_tags)
        for tag in corresponding_moment_tags:
            moments += get_moment_source_lookup(tag)
        processed_moments = format_moment_contents(moments)
        if len(processed_moments) > limit:
            processed_moments = processed_moments[:limit]
    except:
        return []

    return processed_moments


def search_moment_by_dialog_type_full(dialog_type, limit=10):
    try:
        corresponding_moment_tags = dialog_MOMENT_TAG_MAP.get(dialog_type, [])
        moments = []
        # print(corresponding_moment_tags)
        for tag in corresponding_moment_tags:
            moments += get_moment_source_lookup(tag)
        #processed_moments = format_moment_contents(moments)
        if len(moments) > limit:
            moments = moments[:limit]
    except:
        return []

    return moments


def opinion_mining_filter(raw_tweet):
    result = raw_tweet
    for filter_pattern in MOMENT_OPINION_FILTER_PATTERNS:
        result = re.sub(filter_pattern, ' ', result)

    for remove_pattern in MOMENT_OPINION_REMOVE_PATTERNS:
        if re.search(remove_pattern, result):
            return None
    return result


def sentence_similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def select_moment_opinions(moment):
    try:
        opinion_tweets_list = []
        for moment_tweet in moment['tweets']:
            tweet = opinion_mining_filter_tweets(moment_tweet['tweet_text'])
            if tweet:
                subjectivity_score = abs(TextBlob(tweet).sentiment[
                                         0]) + TextBlob(tweet).sentiment[1]
                if subjectivity_score >= MOMENT_OPINION_SCORE_THRESHOLD and sentence_similar(tweet_title, tweet) <= MOMENT_OPINION_SIMILARITY_THRESHOLD:
                    opinion_tweets_list.append(tweet)
    except:
        return []

    return opinion_tweets_list


def user_utterance_paraphrase(user_utterance):
    replacement_rules = [(r"\bare you\b", "am i"), (r"\bi was\b", "you were"), (r"\bi am\b", "you are"), (r"\bwere you\b", "was i"),
                         (r"\bi\b", "you"), (r"\bmy\b",
                                             "your"), (r"\bmyself\b", "yourself"),
                         (r"\byou\b", "i"), (r"\byour\b", "my"),
                         (r"\bme\b", "you")]

    paraphrase_response = user_utterance
    for rule in replacement_rules:
        if re.search(rule[0], user_utterance):
            paraphrase_response = re.sub(rule[0], rule[1], paraphrase_response)
            break

    return paraphrase_response


def unable_answer_question_error_handle(user_utterance, user_context, sys_info, user_attributes):
    error_handle_response = ""
    detected_noun_phrase = sys_info.get('sys_noun_phrase', [])
    detected_noun_phrase.sort(key=lambda s: len(s.split()))
    detected_noun_phrase.reverse()
    # initilaize returnnlp
    # returnnlp = sys_info['features']['returnnlp']
    # initialize the question type
    detected_question_types = user_context["detected_question_types"]

    # paraphrase the question when the length of the question is less than 10
    # elif there is noun_phrase, paraphrse that noun_phrase
    if len(user_utterance.split()) < 10:
        error_handle_response = user_utterance_paraphrase(user_utterance) + '?'
    elif detected_noun_phrase:
        error_handle_respoinse = user_utterance_paraphrase(
            detected_noun_phrase[0]) + '?'

    backstory_tag_list = ["topic_backstory", "ask_hobby", "ask_preference"]

    # Check if the question is a backstory question
    if any(x in detected_question_types for x in backstory_tag_list):
        unable_handle_backstory_question_response = template_dialogs.utterance(selector='grounding/unable_handle_backstory_question', slots={},
                                                                              user_attributes_ref=user_attributes)
        error_handle_response = ' '.join(
            [error_handle_response, unable_handle_backstory_question_response])

    elif "yes_no_question" in detected_question_types and (re.search(r"\byou\b", user_utterance) or re.search(r"\byour\b", user_utterance)):
        # Try to just say no I am not
        if re.search(r"^do", user_utterance):
            unable_handle_yes_or_no_question_response = "no, i don't."
        elif re.search(r"^can", user_utterance):
            unable_handle_yes_or_no_question_response = "sorry, i can't."
        elif re.search(r"^would", user_utterance):
            unable_handle_yes_or_no_question_response = "no, i wouldn't sorry."
        elif re.search(r"^did", user_utterance):
            unable_handle_yes_or_no_question_response = "no, i did't."
        elif re.search(r"^could", user_utterance):
            unable_handle_yes_or_no_question_response = "sorry, i couldn't."
        else:
            unable_handle_yes_or_no_question_response = "sorry, but no."

        error_handle_response = ' '.join(
            [error_handle_response, unable_handle_yes_or_no_question_response])
    elif "open_question_opinion" in detected_question_types:
        unable_handle_opinion_question_response = template_dialogs.utterance(selector='grounding/unable_handle_opinion_question', slots={},
                                                                            user_attributes_ref=user_attributes)
        error_handle_response = ' '.join(
            [error_handle_response, unable_handle_opinion_question_response])
    else:
        unable_handle_question_general_response = template_dialogs.utterance(selector='grounding/unable_handle_question_general', slots={},
                                                                            user_attributes_ref=user_attributes)
        error_handle_response = ' '.join(
            [error_handle_response, unable_handle_question_general_response])

    # Add a break in the end
    error_handle_response = ' '.join(
        [error_handle_response, "<break time='600ms'/>"])

    return error_handle_response

