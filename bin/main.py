import argparse, os, sys

from sentence_analyzer import SentenceAnalyzer

import warnings
warnings.filterwarnings(action='ignore')


if __name__ == "__main__":
    # python main.py -root_dir ../testset -file_name test.txt -batch_size 30 -save_file result.txt -use_gpu
    parser_main = argparse.ArgumentParser(description="main")
    parser_main.add_argument('-root_dir', type=str, required=True)
    parser_main.add_argument('-file_name', type=str, required=True)
    parser_main.add_argument('-save_file', type=str, default="result.txt")
    parser_main.add_argument('-use_gpu', action='store_true')
    parser_main.add_argument('-batch_size', type=int, default=30)

    opt = parser_main.parse_args()
    root_dir = opt.root_dir
    file_name = opt.file_name
    save_file = opt.save_file
    use_gpu = opt.use_gpu
    batch_size = opt.batch_size

    sentences = []
    try:
        with open(os.path.join(root_dir, file_name), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                sentences.append(line)

    except UnicodeDecodeError:
        print("File Encoding 오류({}) 기본 인코딩은 utf-8입니다.".format(len(fail_file)))
        exit()

    sentence_size = len(sentences)
    print("문장 수: {}".format(sentence_size))

    t = SentenceAnalyzer(batch_size=batch_size, gpu=use_gpu)
    res_morphology = t.morphology_analysis(sentences)
    res_parsing = t.dependency_parsing(res_morphology["result_sentences"], file=os.path.join(root_dir, save_file))

    fail_count = res_morphology["error_number"] + res_parsing["error_number"]
    print("\n분석 실패: {}".format(fail_count))
    print("끝")
