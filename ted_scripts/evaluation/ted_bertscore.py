import os
import sys

from bert_score import BERTScorer

root_path = ".."
method = sys.argv[1]
model_id = sys.argv[2]
cuda_id = sys.argv[3]

language_sequence = ["en", "ar", "he", "ru", "ko", "it", "ja", "zh", "es", "nl", "vi", "tr", "fr", "pl", "ro", "th",
                     "fa", "hr", "cs", "de"]


def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]


writing_list = []
for i, tgt in enumerate(language_sequence):
    tmp_score = BERTScorer(lang=tgt, device=f"cuda:{cuda_id}")
    for j, src in enumerate(language_sequence):
        if i == j: continue
        path_ref = os.path.join(root_path, "ted_scripts", "results", method, str(model_id), f"{src}-{tgt}.r")
        path_hypo = os.path.join(root_path, "ted_scripts", "results", method, str(model_id), f"{src}-{tgt}.h")
        ref, hypo = _read_txt_strip_(path_ref), _read_txt_strip_(path_hypo)
        P, R, F = tmp_score.score(hypo, ref, batch_size=100)
        P, R, F = round(P.mean().item() * 100, 2), round(R.mean().item() * 100, 2), round(F.mean().item() * 100, 2)
        print(f"{src}-{tgt}")
        print(f"P: {P} R: {R} F: {F}")
        writing_list.append(f"{src}-{tgt}\n")
        writing_list.append(f"P: {P} R: {R} F: {F} \n")

file = open(
    os.path.join(root_path, "ted_scripts", "results", str(method), str(model_id), f"{str(model_id)}.bertscore"),
    'w', encoding='utf-8')
file.writelines(writing_list)
file.close()
