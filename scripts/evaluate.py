import argparse, json, os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def evaluate(pred_dir, test_json):
    with open(test_json,'r',encoding='utf8') as f:
        tests = json.load(f)
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    results=[]
    for item in tests:
        vid = item['video_id']
        ref = item.get('caption') or item.get('captions')
        if isinstance(ref, list):
            reftxt = " ".join(ref)
        else:
            reftxt = str(ref)
        pred_file = os.path.join(pred_dir, vid, "summary.txt")
        if not os.path.exists(pred_file):
            # fallback: check outputs/<vid>_summary/summary.txt or outputs/<vid>/summary.txt
            continue
        pred = open(pred_file,'r',encoding='utf8').read().strip()
        bleu = sentence_bleu([reftxt.split()], pred.split()) if len(pred.split())>0 else 0.0
        rouge = scorer.score(reftxt, pred)
        results.append((vid, bleu, rouge['rouge1'].fmeasure, rouge['rougeL'].fmeasure))
    # print average
    if len(results)==0:
        print("No predictions found for evaluation.")
        return
    avg_bleu = sum(r[1] for r in results)/len(results)
    avg_r1 = sum(r[2] for r in results)/len(results)
    avg_rl = sum(r[3] for r in results)/len(results)
    print(f"Evaluated {len(results)} videos. Avg BLEU: {avg_bleu:.4f}, ROUGE-1: {avg_r1:.4f}, ROUGE-L: {avg_rl:.4f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--pred_dir", default="outputs")
    p.add_argument("--test_json", default="dataset/msrvtt_test_1k.json")
    args=p.parse_args()
    evaluate(args.pred_dir, args.test_json)
