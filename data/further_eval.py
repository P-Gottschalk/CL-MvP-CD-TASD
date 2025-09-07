'''
The code found in this page is to calculate more extensive scores from already available predictions. We distinguish between Model and LLM data.
Part of the code is inspired by https://github.com/ZubinGou/multi-view-prompting.git
'''

import argparse
import os
import json
import pandas as pd
import sys
import ast

sys.path.append('/content/Master_Thesis')

from model.eval_utils import compute_f1_scores

def create_f1_indiv_cat(args, which_type):
    suffix_map = {
        "best_ckpt": "-ckpt.json",
        "last_5":   "-5ckpt.json",
        "full_model": ".json",
    }
    try:
        suffix = suffix_map[which_type]
    except KeyError:
        raise ValueError(f"Unknown type: {which_type!r}")
    
    columns = ["Full Model", "MVP only", "CL only", "Base MT5"]
    df = pd.DataFrame(columns=columns)

    all_languages = list(args.languages) + ["English"]

    def process_mode(mode_name, langs):
        for col in columns:
            mvp = 5 if col in ("Full Model", "MVP only") else 1
            contr = "-sent-aspect" if col in ("Full Model", "CL only") else ""

            for lang in langs:
                if mode_name == "Unilingual":
                    folder_prefix = lang
                elif mode_name == "Cross-Lingual-ENG":
                    folder_prefix = "English"
                else:
                    folder_prefix = mode_name

                for extra in ("", "-CD"):
                    filename = (f"{folder_prefix}_{args.dataset}/"f"results-{lang}-{args.dataset}-"f"{args.num_train_epochs}epochs-"f"{mvp}mvp-{args.weight_decay}-"f"{args.warmup_steps}-"f"{args.hidden_layers}{extra}{contr}"f"{suffix}")

                    full_path = os.path.join(args.path, filename)
                    row_idx = f"{mode_name}{extra}: {lang}"

                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding="utf-8") as f:
                            data = json.load(f)
                        output = data.get("examples")

                        all_labels_ac, all_preds_ac, all_labels_at, all_preds_at, all_labels_sp, all_preds_sp = [], [], [], [], [], []

                        for example in output:
                            gold_list_ac, pred_list_ac, gold_list_at, pred_list_at, gold_list_sp, pred_list_sp = [], [], [], [], [], []

                            labels = example.get("labels_correct")
                            for label in labels:
                                ac = label[0]
                                at = label[1]
                                sp = label[2]                    

                                gold_list_ac.append(ac)
                                gold_list_at.append(at)
                                gold_list_sp.append(sp)
                            
                            labels = example.get("labels_pred")
                            for label in labels:
                                try:
                                    ac = label[0]
                                    at = label[1]
                                    sp = label[2]

                                    pred_list_ac.append(ac)
                                    pred_list_at.append(at)
                                    pred_list_sp.append(sp)
                                except Exception as e:
                                    print(f"Parsing error: {e}")
                            
                            all_labels_ac.append(gold_list_ac)
                            all_labels_at.append(gold_list_at)
                            all_labels_sp.append(gold_list_sp)

                            all_preds_ac.append(pred_list_ac)
                            all_preds_at.append(pred_list_at)
                            all_preds_sp.append(pred_list_sp)
                        
                        scores_ac = compute_f1_scores(all_preds_ac, all_labels_ac, indiv=True)
                        scores_at = compute_f1_scores(all_preds_at, all_labels_at, indiv=True)
                        scores_sp = compute_f1_scores(all_preds_sp, all_labels_sp, indiv=True)

                        df.loc[f"{row_idx}_aspect", col] = scores_at.get("f1", None)
                        df.loc[f"{row_idx}_category", col] = scores_ac.get("f1", None)
                        df.loc[f"{row_idx}_sentiment", col] = scores_sp.get("f1", None)
                    else:
                        df.loc[f"{row_idx}_aspect", col] = None
                        df.loc[f"{row_idx}_category", col] = None
                        df.loc[f"{row_idx}_sentiment", col] = None 

    process_mode("Unilingual", all_languages)

    if args.multilingual:
        process_mode("Multilingual", all_languages)

    if args.multilingual_limited:
        process_mode("Multilingual_Limited", all_languages)

    if args.multilingual_small:
        process_mode("Multilingual_Small", all_languages)

    if args.multilingual_small_limited:
        process_mode("Multilingual_Small_Limited", all_languages)

    if args.english:
        process_mode("Cross-Lingual-ENG", args.languages)

    return df.sort_index()

def create_f1_indiv_llm(args):
    all_languages = list(args.languages) + ["English"]

    columns = ["zero-shot-template", "zero-shot-cot-template", "few-shot-template", "few-shot-cot-template", "Fine-tuned"]
    df = pd.DataFrame(columns=columns)

    def process_mode(mode_name, langs):
        for col in columns:
            for lang in langs: 

                if mode_name == "Unilingual":
                    prefix = lang
                    row_idx = f"{mode_name}_{lang}"
                elif mode_name == "Cross-lingual":
                    prefix = "English"
                    row_idx = f"{mode_name}_{lang}"
                elif mode_name == "Multilingual_Small":
                    prefix = "Multilingual"
                    row_idx = f"{prefix}_{lang}"
                elif mode_name == "Multilingual_Small_Limited":
                    prefix = "Multilingual_Limited"
                    row_idx = f"0_{prefix}_{lang}"

                path_to_results = f"{args.path_llm}/{args.dataset}_Restaurants_Generated_{prefix}_{lang}_{col}.txt"

                if os.path.exists(path_to_results):
                    all_labels_ac, all_preds_ac, all_labels_at, all_preds_at, all_labels_sp, all_preds_sp = [], [], [], [], [], []
                    gold_list_ac, pred_list_ac, gold_list_at, pred_list_at, gold_list_sp, pred_list_sp = [], [], [], [], [], []

                    with open(path_to_results,'r',encoding='utf-8') as file:
                        for line in file:
                            if line.startswith("Sentiment Elements:"):
                                line = line.split("Sentiment Elements:")[1].strip()

                                try:
                                    pred_list = ast.literal_eval(line)

                                    for t in pred_list:
                                        pred_list_at.append(t[0])
                                        pred_list_ac.append(t[1])
                                        pred_list_sp.append(t[2])  
                                except Exception as e:
                                    print(f"Parsing error: {e}")
                                    pred_list_at, pred_list_ac, pred_list_sp = [], [], []
                            elif line.startswith("Gold:"):
                                line = line.split("Gold:")[1].strip()
                                gold_list = ast.literal_eval(line)

                                for t in gold_list:
                                    gold_list_at.append(t[0])
                                    gold_list_ac.append(t[1])
                                    gold_list_sp.append(t[2])

                                all_labels_at.append(gold_list_at.copy())
                                all_preds_at.append(pred_list_at.copy())
                                all_labels_ac.append(gold_list_ac.copy())
                                all_preds_ac.append(pred_list_ac.copy())
                                all_labels_sp.append(gold_list_sp.copy())
                                all_preds_sp.append(pred_list_sp.copy())
                                
                                gold_list_ac, pred_list_ac, gold_list_at, pred_list_at, gold_list_sp, pred_list_sp = [], [], [], [], [], []

                        scores_at = compute_f1_scores(all_preds_at, all_labels_at, indiv=True)
                        scores_ac = compute_f1_scores(all_preds_ac, all_labels_ac, indiv=True)
                        scores_sp = compute_f1_scores(all_preds_sp, all_labels_sp, indiv=True)

                        df.loc[f"{row_idx}_aspect", col] = scores_at.get("f1", None)
                        df.loc[f"{row_idx}_category", col] = scores_ac.get("f1", None)
                        df.loc[f"{row_idx}_sentiment", col] = scores_sp.get("f1", None)
                else:
                    df.loc[f"{row_idx}_aspect", col] = None
                    df.loc[f"{row_idx}_category", col] = None
                    df.loc[f"{row_idx}_sentiment", col] = None

    process_mode("Unilingual", all_languages)
    process_mode("Multilingual_Small", all_languages)
    if args.dataset == "MABSA": process_mode("Multilingual_Small_Limited", all_languages)
    process_mode("Cross-lingual", args.languages)

    return df.sort_index()

def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default="/content/drive/MyDrive/data/models", help="The path to read the data from")
    parser.add_argument("--languages", nargs="+", default=["French", "Spanish", "Dutch", "Turkish"], help="List of languages")

    parser.add_argument("--full_model", action='store_true', help='Include the full model results')
    parser.add_argument("--last_5", action='store_true', help='Include the results from the last 5 epochs')
    parser.add_argument("--best_ckpt", action='store_true', help='Include the results from the best checkpoint')

    parser.add_argument("--english", action='store_true', help='English models have been trained')
    parser.add_argument("--multilingual", action='store_true', help='Multilingual models have been trained')
    parser.add_argument("--multilingual_limited", action='store_true', help='Multilingual limited models have been trained')
    parser.add_argument("--multilingual_small", action='store_true', help='Multilingual small models have been trained')
    parser.add_argument("--multilingual_small_limited", action='store_true', help='Multilingual small limited models have been trained')

    parser.add_argument("--hidden_layers", default=1024, type=int, help="Size of the Seq2Seq model")
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=0.1, type=float)
    parser.add_argument("--dataset", type=str, default="SemEval16",  help="Which dataset you want to use")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")

    parser.add_argument("--llm", action='store_true', help="Create tables for LLMs")
    parser.add_argument("--path_llm", type=str, default="/content/drive/MyDrive/data/llm", help="The path to the LLM folder")

    args = parser.parse_args()

    output_dir = os.path.join(args.path, "combined_results/extended")
    os.makedirs(output_dir, exist_ok=True)

    if not args.llm:
        for flag, which in [(args.full_model, "full_model"), (args.last_5, "last_5"), (args.best_ckpt, "best_ckpt")]:
            if flag:
                df = create_f1_indiv_cat(args, which)
                out_file = os.path.join(output_dir, f"results-{args.dataset}-{args.num_train_epochs}-{args.hidden_layers}-{args.weight_decay}-{args.warmup_steps}-{which}-individual_cat.xlsx")
                df.to_excel(out_file, index=True)

                print(f"Wrote results to {out_file}")
    else:
        df = create_f1_indiv_llm(args)
        out_file = os.path.join(output_dir, f"results-{args.dataset}-LLM-individual_cat.xlsx")
        df.to_excel(out_file, index=True)

        print(f"Wrote results to {out_file}")

if __name__ == "__main__":
    main()
