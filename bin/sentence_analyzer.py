import math, os, json, re, sys
import torch
import torch.utils.data
import argparse

from sentence_process import SentenceProcessing

from dataset import collate_fn, TranslationDataset
import transformer.Constants as Constants
from transformer.Translator import Translator

import numpy as np
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.models import StackPtrNet
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

from collections import OrderedDict


class SentenceAnalyzer(object):
    def __init__(self, morphology_analysis="./models/trained.chkpt",
                 morphology_vocab="./models/sejong.pt",
                 dependency_parser_path="./models",
                 dependency_parser_name="DependencyParser.pt",
                 debug=False, system=False, batch_size=30, gpu=False):
        self.debug = debug
        self.system = system
        self.gpu = gpu
        self.batch_size = batch_size
        self.sentence_processing = SentenceProcessing(debug=self.debug)
        self._init_morphology_analysis_for_system(morphology_analysis, morphology_vocab)
        self._init_dependency_parsing(dependency_parser_path, dependency_parser_name)

    def _init_morphology_analysis_for_system(self, model, vocab):
        parser_m = argparse.ArgumentParser(prog="sentence_analyzer", description='translate.py')
        parser_m.add_argument('-model', default=model,
                            help='Path to model .pt file')
        parser_m.add_argument('-vocab', default=vocab,
                            help='Source sequence to decode (one line per sequence)')
        parser_m.add_argument('-beam_size', type=int, default=5,
                            help='Beam size')
        parser_m.add_argument('-batch_size', type=int, default=self.batch_size,
                            help='Batch size')
        parser_m.add_argument('-n_best', type=int, default=1,
                            help="""If verbose is set, will output the n_best
                                        decoded sentences""")
        parser_m.add_argument('-no_cuda', action='store_true')

        # dummy
        parser_m.add_argument('-root_dir', type=str, default="")
        parser_m.add_argument('-file_name', type=str, default="")
        parser_m.add_argument('-save_file', type=str, default="")
        parser_m.add_argument('-use_gpu', action='store_true')

        opt = parser_m.parse_args()
        opt.cuda = self.gpu

        # Prepare DataLoader
        self.preprocess_data = torch.load(opt.vocab)
        self.preprocess_settings = self.preprocess_data['settings']

        self.morphology_analyzer = Translator(opt)

    def _init_dependency_parsing(self, model_path, model_name):
        args_parser = argparse.ArgumentParser(description='Testing with stack pointer parser')

        args_parser.add_argument('--model_path', help='path for parser model directory', default=model_path)
        args_parser.add_argument('--model_name', help='parser model file', default=model_name)
        args_parser.add_argument('--test', default="tmp.txt")
        args_parser.add_argument('--beam', type=int, default=5, help='Beam size for decoding')
        args_parser.add_argument('--batch_size', type=int, default=self.batch_size)

        # dummy
        args_parser.add_argument('-root_dir', type=str, default="")
        args_parser.add_argument('-file_name', type=str, default="")
        args_parser.add_argument('-save_file', type=str, default="")
        args_parser.add_argument('-use_gpu', action='store_true')
        args_parser.add_argument('-batch_size', type=int, default=self.batch_size)

        args = args_parser.parse_args()

        model_path = args.model_path
        model_name = os.path.join(model_path, args.model_name)
        self.beam = args.beam

        def load_args():
            with open("{}.arg.json".format(model_name)) as f:
                key_parameters = json.loads(f.read())

            return key_parameters['args'], key_parameters['kwargs']

        arguments, kwarguments = load_args()
        mode = arguments[10]
        input_size_decoder = arguments[11]
        hidden_size = arguments[12]
        arc_space = arguments[16]
        type_space = arguments[17]
        encoder_layers = arguments[13]
        decoder_layers = arguments[14]
        char_num_filters = arguments[6]
        eojul_num_filters = arguments[8]
        p_rnn = kwarguments['p_rnn']
        p_in = kwarguments['p_in']
        p_out = kwarguments['p_out']
        self.prior_order = kwarguments['prior_order']
        skipConnect = kwarguments['skipConnect']
        grandPar = kwarguments['grandPar']
        sibling = kwarguments['sibling']
        use_char = kwarguments['char']
        use_pos = kwarguments['pos']
        use_eojul = kwarguments['eojul']

        alphabet_path = os.path.join(model_path, 'alphabets/')
        self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet = conllx_stacked_data.load_alphabets(alphabet_path)
        num_words = self.word_alphabet.size()
        num_chars = self.char_alphabet.size()
        num_pos = self.pos_alphabet.size()
        num_types = self.type_alphabet.size()

        word_table = None
        word_dim = arguments[0]
        char_table = None
        char_dim = arguments[2]
        pos_table = None
        pos_dim = arguments[4]

        char_window = arguments[7]
        eojul_window = arguments[9]

        self.dependency = StackPtrNet(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, char_num_filters, char_window, eojul_num_filters, eojul_window,
                                      mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers,
                                      num_types, arc_space, type_space,
                                      embedd_word=word_table, embedd_char=char_table, embedd_pos=pos_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                                      biaffine=True, pos=use_pos, char=use_char, eojul=use_eojul, prior_order=self.prior_order,
                                      skipConnect=skipConnect, grandPar=grandPar, sibling=sibling, gpu=self.gpu)

        if self.gpu:
            self.dependency.cuda()

        if self.gpu:
            self.dependency.load_state_dict(torch.load(model_name))
        else:
            self.dependency.load_state_dict(torch.load(model_name, map_location='cpu'))
        self.dependency.eval()

    def morphology_analysis(self, sentences):
        org_sentences, org_sentence_space_infos, input_sentences, sentence_symbol_mappings = self.sentence_processing.input_conversion_sentences(sentences)
        if self.system:
            sent = org_sentences[0]
            if len(sent.split(" ")) >= 50:
                error_code = 12
                error_msg = "문장이 길어 분석을 수행할 수 없습니다."
                return {"error_code":error_code, "error_msg": error_msg}
        err_code, err_msg, word_insts = self.sentence_processing.get_instances_for_morphology(input_sentences, self.preprocess_settings.max_word_seq_len, system=self.system)
        if err_code != -1:
            return {"error_code":err_code, "error_msg":err_msg}

        word_insts, sentence_symbol_mappings = self.sentence_processing.convert_instance_to_idx_seq_for_morphology(word_insts, self.preprocess_data['dict']['src'], sentence_symbol_mappings)

        data_loader = torch.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=self.preprocess_data['dict']['src'],
                tgt_word2idx=self.preprocess_data['dict']['tgt'],
                src_insts=word_insts),
            num_workers=2,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )
        all_output_sentences = []

        if not self.system:
            total_batch = len(data_loader)

        for idx, batch in enumerate(data_loader):
            if not self.system:
                sys.stdout.write("\r형태소 분석 중: {}/{}".format(idx+1, total_batch))
                sys.stdout.flush()

            all_hyp, all_scores = self.morphology_analyzer.translate_batch(*batch)
            for idx_seqs in all_hyp:
                nbest_sentences = []
                for idx_seq in idx_seqs:
                    pred_line = " ".join([data_loader.dataset.tgt_idx2word[idx] for idx in idx_seq][:-1])
                    if self.debug:
                        print("Morphology Character Output Lengths: {}".format(len(pred_line.split(" "))))
                    nbest_sentences.append(pred_line)
                all_output_sentences.append(nbest_sentences)

        output_sentences = [x[0] for x in all_output_sentences]
        restore_symbol_output_sentences = self.sentence_processing.convert_key_to_symbol(output_sentences, sentence_symbol_mappings)
        err_code, err_msg, convert_output_sentences, not_convert_output_sentences = self.sentence_processing.character_to_morphology(input_sentences, restore_symbol_output_sentences, system=self.system)
        if err_code != -1:
            return {"error_code":err_code, "error_msg":err_msg}

        if self.debug:
            for org_sentence, org_sentence_space_info, input_sentence, sentence_symbol_mapping, output_sentence in zip(org_sentences, org_sentence_space_infos, input_sentences, sentence_symbol_mappings, convert_output_sentences):
                print("ORG sentence: {}".format(org_sentence))
                print("ORG sentence space information: {}".format(org_sentence_space_info))
                print("Input Sentence: {}".format(input_sentence))
                print("Sentence Symbol Mappings: {}".format(sentence_symbol_mapping.items()))
                if output_sentence is not None:
                    print("Output Sentence: {}".format(output_sentence))
                    print()
                else:
                    print("Output Sentence: ERROR!")
                    print()

        conll_of_sentences = self.sentence_processing.convert_to_CoNLL(sentences, org_sentences, org_sentence_space_infos, convert_output_sentences)

        if self.debug:
            for sentence in conll_of_sentences:
                print("\n".join(sentence))
                print("\n")

        return {"error_code": err_code, "error_msg": err_msg, "error_number": len(not_convert_output_sentences), "result_sentences": conll_of_sentences}

    def check_crossing(self, headers):
        result = True
        id_header_pairs = [(idx+1, head[6]) for idx, head in sorted(headers.items(), key=lambda x:x[0])]
        for i, (idx, head_number) in enumerate(id_header_pairs):
            for idx_, head_number_ in id_header_pairs[i+1:]:
                gap = math.fabs(idx - head_number)
                if gap == 1:
                    break

                if idx < idx_ and idx_ < head_number:
                    gap2 = math.fabs(idx_ - head_number_)
                    if gap < gap2 and head_number_ != 0:
                        result = False
                        break

            if not result:
                break

        return result

    def check_cycle(self, headers):
        result = True
        for idx in sorted(headers.keys()):
            visit_idx = OrderedDict()
            while headers[idx][6] != 0:
                if headers[idx][6] - 1 in visit_idx:
                    result = False
                    break
                idx = headers[idx][6] - 1
                visit_idx.setdefault(idx)

        return result


    def dependency_parsing(self, input_sentences, file=None):
        error_code = -1
        error_msg = ""
        pred_writer = CoNLLXWriter(self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet)

        datas = conllx_stacked_data.read_stacked_data_to_variable(input_sentences, self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet, use_gpu=self.gpu, prior_order=self.prior_order)
        parsing_datas = []

        if not self.system:
            print("")
            total_datas = np.sum(np.array(datas[1]))
            cnt_datas = 0

        for batch in conllx_stacked_data.iterate_batch_stacked_variable(datas, self.batch_size, use_gpu=self.gpu):
            input_encoder, _, sentences, comments = batch

            word, char, pos, heads, types, masks, lengths = input_encoder

            if not self.system:
                cnt_datas += len(sentences)
                sys.stdout.write("\r의존구문 분석 중: {}/{}".format(cnt_datas, total_datas))
                sys.stdout.flush()

            heads_pred, types_pred, _, _ = self.dependency.decode(word, char, pos, mask=masks, length=lengths, beam=self.beam, leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()

            parsing_datas.extend(pred_writer.overwrite(sentences, comments, word, pos, heads_pred, types_pred, lengths, symbolic_root=True))

        clear_parsing_datas = []
        not_clear_parsing_datas = []

        for comment, parsing_result in parsing_datas:
            headers = {data[0]-1:data for data in parsing_result}
            iscycle = self.check_cycle(headers)
            iscrossing = self.check_crossing(headers)
            if not iscycle or not iscrossing:
                not_clear_parsing_datas.append((comment, parsing_result))
            else:
                clear_parsing_datas.append((comment, parsing_result))

        if not self.system:
            pred_writer.start(file)
            pred_writer.batch_write(clear_parsing_datas)
            pred_writer.close()

        if self.system and len(not_clear_parsing_datas) > 0:
            error_code = 21
            error_msg = "구문 분석을 수행할 수 없습니다.(Error Code: {})".format(error_code)
            return {"error_code": error_code, "error_msg": error_msg}

        return {"error_code": error_code, "error_msg": error_msg, "error_number": len(not_clear_parsing_datas), "result_sentences": clear_parsing_datas}

    def process(self, sentence):
        error_code = -1
        error_msg = ""
        sentence = sentence.strip()
        sentences = [sentence]
        res_morphology = self.morphology_analysis(sentences)
        if res_morphology["error_code"] != -1:
            return res_morphology
        res_parsing = self.dependency_parsing(res_morphology["result_sentences"])

        if res_parsing["error_code"] != -1:
            return res_parsing

        if self.debug:
            for comment, result_parsing in res_parsing["result_sentences"]:
                print("")
                print("\n".join(comment))
                for line in result_parsing:
                    print("\t".join([str(x) for x in line]))
                print("")

        simplify_sentence = self.sentence_processing.simplify(res_parsing["result_sentences"][0][1])

        result = {key:value for key, value in simplify_sentence.items()}
        result.setdefault("error_code", error_code)
        result.setdefault("error_msg", error_msg)
        result.setdefault("conll", res_parsing["result_sentences"][0][1])
        return result


