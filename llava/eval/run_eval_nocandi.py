import argparse
import json
import collections
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from eval_metrics.glossary import *

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to ground truth file')
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file')
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def align_dict_lists(gt, pred):
    gt_ids = {item['id'] for item in gt}
    pred_ids = {item['question_id'] for item in pred}

    common_ids = gt_ids & pred_ids

    gt_aligned = [item for item in gt if item['id'] in common_ids]
    pred_aligned = [item for item in pred if item['question_id'] in common_ids]

    gt_aligned.sort(key=lambda x: x['id'])
    pred_aligned.sort(key=lambda x: x['question_id'])

    return gt_aligned, pred_aligned

def evaluate(gt, pred):    
    gt, pred = align_dict_lists(gt, pred)
    assert len(gt) == len(pred), "the length of gt is the same as pred"
    scores = collections.defaultdict(list)
    closed_scores = collections.defaultdict(list)
    closed_questions_count = 0
    closed_questions_correct = 0
    for gt_item, pred_item in zip(gt, pred):
        gt_results = gt_item.get('conversations', gt_item.get('conversatons'))
        gt_value = gt_results[1]['value'].lower()
        pred_value = pred_item['text'].lower()
        answer_type = gt_item['answer_type'] if 'answer_type' in gt_item else 'OPEN'

        # Your normalization function here
        # gt_value = normalize_word(gt_value)
        # pred_value = normalize_word(pred_value)
        print(answer_type)
        if answer_type == 'OPEN':
            # Assuming calculate_exactmatch, calculate_f1score functions are defined elsewhere
            scores['exact_match'].append(calculate_exactmatch(pred_value, gt_value))
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            scores['f1'].append(f1_score)
            scores['precision'].append(precision)
            scores['recall'].append(recall)

            # Calculate BLEU scores with different weights
            weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]  # You can customize the weights here
            bleu_scores = [sentence_bleu([gt_value.split()], pred_value.split())]
            for w in weights:
                bleu_score = sentence_bleu([gt_value.split()], pred_value.split(), weights=w, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu_score)
            scores['bleu_scores'].append(bleu_scores)
            

        #calculate 'yes/no accuracy'
        elif answer_type == 'CLOSED':
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            closed_scores['f1'].append(f1_score)
            closed_scores['precision'].append(precision)
            closed_scores['recall'].append(recall)
            closed_questions_count += 1
            if gt_value in pred_value:  # Check if gt_value is contained within pred_value
                closed_questions_correct += 1


    # Calculate average scores
    print(scores)
    exact_match_avg = sum(scores['exact_match']) / len(scores['exact_match'])
    f1_score_avg = sum(scores['f1']) / len(scores['f1'])
    precision_avg = sum(scores['precision']) / len(scores['precision'])
    recall_avg = sum(scores['recall']) / len(scores['recall'])

    # Calculate average BLEU scores for different weights
    bleu_scores_avg = [sum(score_list) / len(score_list) for score_list in zip(*scores['bleu_scores'])]

    # Calculate closed question accuracy
    if 'answer_type' in gt_item:
        print(f"count: {closed_questions_count}, correct: {closed_questions_correct}")
        closed_score = (closed_questions_correct / closed_questions_count) if closed_questions_count else 0
        closed_f1_score_avg = sum(closed_scores['f1']) / len(closed_scores['f1'])
        closed_precision_avg = sum(closed_scores['precision']) / len(closed_scores['precision'])
        closed_recall_avg = sum(closed_scores['recall']) / len(closed_scores['recall'])
    
        # Generate evaluation summary
        results_table = tabulate(
            [
                ['Exact Match Score', exact_match_avg*100],
                ['F1 Score', f1_score_avg*100],
                ['Precision', precision_avg*100],
                ['Recall', recall_avg*100],
                ['BLEU Score', bleu_scores_avg[0]*100], 
                ['BLEU Score (Weight 1)', bleu_scores_avg[1]*100],
                ['BLEU Score (Weight 2)', bleu_scores_avg[2]*100],
                ['BLEU Score (Weight 3)', bleu_scores_avg[3]*100],
                ['yes/no accuracy', closed_score*100],
                ['Closed F1 Score', closed_f1_score_avg*100],
                ['Closed Precision', closed_precision_avg*100],
                ['Closed Recall', closed_recall_avg*100],
            ],
            headers=['Metric', 'Performance (%)']
        )
    else:
        # Generate evaluation summary
        results_table = tabulate(
            [
                ['Exact Match Score', exact_match_avg*100],
                ['F1 Score', f1_score_avg*100],
                ['Precision', precision_avg*100],
                ['Recall', recall_avg*100],
                ['BLEU Score', bleu_scores_avg[0]*100], 
                ['BLEU Score (Weight 1)', bleu_scores_avg[1]*100],
                ['BLEU Score (Weight 2)', bleu_scores_avg[2]*100],
                ['BLEU Score (Weight 3)', bleu_scores_avg[3]*100],
                # ['yes/no accuracy', closed_score*100],
                # ['Closed F1 Score', closed_f1_score_avg*100],
                # ['Closed Precision', closed_precision_avg*100],
                # ['Closed Recall', closed_recall_avg*100],
            ],
            headers=['Metric', 'Performance (%)']
        )
    return results_table

if __name__ == '__main__':
    args = parse_option()

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)
    results = evaluate(gt, pred)
    print(results)
