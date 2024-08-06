import os

lang_list = [
    "en", "ar", "he", "ru", "ko", "it", "ja",
    "zh", "es", "nl", "vi", "tr", "fr", "pl",
    "ro", "fa", "hr", "cs", "de",
]


def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]


def _write_text_(url, data):
    file = open(url, 'w', encoding='utf-8')
    for i in data:
        file.write(i + "\n")
    file.close()


dir_path = "../ted_scripts/build_data/turncated"
os.makedirs(dir_path, exist_ok=True)
for i in lang_list:
    for j in lang_list:
        if i == j: continue
        zeroshot_flag = True if i != "en" and j != "en" else False
        os.makedirs(os.path.join(dir_path, f"{i}_{j}"), exist_ok=True)
        test_src = _read_txt_strip_(os.path.join("../ted_scripts/build_data/raw", f"{i}_{j}/test.{i}"))
        test_tgt = _read_txt_strip_(os.path.join("../ted_scripts/build_data/raw", f"{i}_{j}/test.{j}"))
        _write_text_(os.path.join(dir_path, f"{i}_{j}/test.{i}"), test_src[:2000])
        _write_text_(os.path.join(dir_path, f"{i}_{j}/test.{j}"), test_tgt[:2000])
        if not zeroshot_flag:
            valid_src = _read_txt_strip_(os.path.join("../ted_scripts/build_data/raw", f"{i}_{j}/dev.{i}"))
            valid_tgt = _read_txt_strip_(os.path.join("../ted_scripts/build_data/raw", f"{i}_{j}/dev.{j}"))
            _write_text_(os.path.join(dir_path, f"{i}_{j}/valid.{i}"), valid_src[:2000])
            _write_text_(os.path.join(dir_path, f"{i}_{j}/valid.{j}"), valid_tgt[:2000])
