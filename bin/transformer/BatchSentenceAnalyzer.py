import math, os, json, re
import torch
import torch.utils.data
import argparse

from dataset import collate_fn, TranslationDataset
import transformer.Constants as Constants
from transformer.Translator import Translator
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.models import StackPtrNet
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

from UPosTagMap import *
from SimplifyDepParse3 import SimplifyDepParse

class SentenceAnalyzer(object):
    def __init__(self, morphology_analysis="./models/MorphologicalAnalyzer.chkpt", morphology_vocab="./models/MorphologicalAnalyzerVocab.pt", dependency_parser_path="./models", dependency_parser_name="DependencyParser.pt"):
        self._init_morphology_analysis(morphology_analysis, morphology_vocab)
        self._init_dependency_parser(dependency_parser_path, dependency_parser_name)
        self.upostag_map = UPosTagMap()
        self.oo = SimplifyDepParse()

    def _init_morphology_analysis(self, model, vocab):
        parser = argparse.ArgumentParser(description='translate.py')

        parser.add_argument('-model', default=model,
                            help='Path to model .pt file')
        parser.add_argument('-vocab', default=vocab,
                            help='Source sequence to decode (one line per sequence)')
        parser.add_argument('-beam_size', type=int, default=5,
                            help='Beam size')
        parser.add_argument('-batch_size', type=int, default=1,
                            help='Batch size')
        parser.add_argument('-n_best', type=int, default=1,
                            help="""If verbose is set, will output the n_best
                                decoded sentences""")
        parser.add_argument('-no_cuda', action='store_true')

        opt = parser.parse_args()
        opt.cuda = not opt.no_cuda

        # Prepare DataLoader
        self.preprocess_data = torch.load(opt.vocab)
        self.preprocess_settings = self.preprocess_data['settings']

        self.translator = Translator(opt)

    def _init_dependency_parser(self, model_path="", model_name=""):
        args_parser = argparse.ArgumentParser(description='Testing with stack pointer parser')

        args_parser.add_argument('--model_path', help='path for parser model directory', default=model_path)
        args_parser.add_argument('--model_name', help='parser model file', default=model_name)
        args_parser.add_argument('--test', default="tmp.txt")
        args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
        args_parser.add_argument('--batch_size', type=int, default=1)

        args = args_parser.parse_args()

        model_path = args.model_path
        model_name = os.path.join(model_path, args.model_name)
        self.test_path = args.test
        beam = args.beam
        batch_size = args.batch_size

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
                              skipConnect=skipConnect, grandPar=grandPar, sibling=sibling)

        self.dependency.load_state_dict(torch.load(model_name, map_location='cpu'))
        self.dependency.eval()

    def transform_sentence(self, sentence):
        tokens = sentence.split(" ")

        result = []
        result_org = []
        org_space = []
        for token in tokens:
            result_tmp_tokens = []
            if "[" in token and "]" not in token:
                tmp_token = token.replace("[", " [ ")
                tmp_tokens = [x for x in tmp_token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            elif "[" not in token and "]" in token:
                tmp_token = token.replace("]", " ] ")
                tmp_tokens = [x for x in tmp_token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            elif "(" in token and ")" not in token:
                tmp_token = token.replace("(", " ( ")
                tmp_tokens = [x for x in tmp_token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            elif "(" not in token and ")" in token:
                tmp_token = token.replace(")", " ) ")
                tmp_tokens = [x for x in tmp_token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            elif "{" in token and "}" not in token:
                tmp_token = token.replace("{", " { ")
                tmp_tokens = [x for x in tmp_token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            elif "}" not in token and "}" in token:
                tmp_token = token.replace("}", " } ")
                tmp_tokens = [x for x in tmp_token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            elif token.count("\'") == 1:
                tmp_token = token.replace("\'", " \' ")
                tmp_tokens = [x for x in tmp_token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            elif token.count("\"") == 1:
                tmp_token = token.replace("\"", " \" ")
                tmp_tokens = [x for x in tmp_token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            else:
                result_tmp_tokens.append(token)

            reuslt_tmp_tokens_length = len(result_tmp_tokens)

            for idx, t in enumerate(result_tmp_tokens, start=1):
                result.extend(t)
                result_org.append(t)
                if idx == reuslt_tmp_tokens_length:
                    org_space.append(False)
                else:
                    org_space.append(True)

                result.append("<sa>")

        return " ".join(result[:-1]), " ".join(result_org), org_space

    def get_instances(self, sentence, max_sent_len, keep_case):
        word_insts = []
        trimmed_sent_count = 0
        if not keep_case:
            sentence = sentence.lower()
        words = sentence.split()
        if len(words) > max_sent_len:
            trimmed_sent_count += 1
        word_inst = words[:max_sent_len]

        if word_inst:
            word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
        else:
            word_insts += [None]
        return word_insts

    def convert_instance_to_idx_seq(self, word_insts, word2idx):
        ''' Mapping words to idx sequence. '''
        return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

    def transform_result(self, sentence):
        result = []

        tokens = sentence.split(" ")
        m_tmp = []
        for token in tokens:
            if token == "<sa>":
                result.append(token)
            elif re.search("<[a-z]+>", token):
                morph = "".join(m_tmp)
                pos = token.upper().replace("<", "").replace(">", "")
                tok = "{}/{}".format(morph, pos)
                result.append(tok)
                m_tmp = []
            else:
                m_tmp.append(token)

        return " ".join(result)

    def convertToCoNLL(self, org_sentence, org_space_info, sentence):
        org_tokens = [x.strip() for x in org_sentence.split(" ")]
        tokens = [x.strip() for x in sentence.split("<sa>")]
        conll = []

        for idx, (token, org_token, space_info) in enumerate(zip(tokens, org_tokens, org_space_info) , start=1):
            only_mo = []
            only_pos = []
            for mo in token.split(" "):
                m, p = mo.split("/")
                if m == "<sf>":
                    m = "."
                only_mo.append(m)
                only_pos.append(p)
            input_xpostag = " ".join(only_pos)
            lemma = " ".join(only_mo)
            if space_info:
                tmp = [str(idx), org_token, " ".join(only_mo), "+".join(self.upostag_map.GetUPOS(input_xpostag, lemma)),
                       "+".join(only_pos), "_", "0", "root", "_", "SpaceAfter=No"]
            else:
                tmp = [str(idx), org_token, " ".join(only_mo), "+".join(self.upostag_map.GetUPOS(input_xpostag, lemma)), "+".join(only_pos), "_", "0", "root", "_", "_"]
            conll.append("\t".join(tmp))
        return "\n".join(conll)

    def morphologyAnalysis(self, sentence):
        sentence, org_sentence, org_space_info = self.transform_sentence(sentence)
        token_length = len(org_sentence.split(" "))
        test_src_word_insts = self.get_instances(sentence, self.preprocess_settings.max_word_seq_len, self.preprocess_settings.keep_case)
        test_src_insts = self.convert_instance_to_idx_seq(test_src_word_insts, self.preprocess_data['dict']['src'])

        test_loader = torch.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=self.preprocess_data['dict']['src'],
                tgt_word2idx=self.preprocess_data['dict']['tgt'],
                src_insts=test_src_insts),
            num_workers=2,
            batch_size=1,
            collate_fn=collate_fn)

        for batch in test_loader:
            all_hyp, all_scores = self.translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = " ".join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])


        result_line = self.transform_result(pred_line)
        result_length = len(result_line.split(" <sa> "))
        print(org_sentence.split(" "))
        print(result_line.split(" <sa> "))
        if token_length == result_length:
            return self.convertToCoNLL(org_sentence, org_space_info, result_line)

        return result_line

    def convert(self, sentences, word, pos, head, type, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _, lemma_length = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        result = []
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                if len(sentences[i]) == j-start:
                    break

                t = self.type_alphabet.get_instance(type[i, j])
                h = head[i, j]

                result.append('%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' % (j, sentences[i][j-start][1], sentences[i][j-start][2], sentences[i][j-start][3], \
                                                                                       sentences[i][j-start][4], sentences[i][j-start][5], h, t, sentences[i][j-start][8], sentences[i][j-start][9]))
        return result

    def dependencyparsing(self):
        pred_writer = CoNLLXWriter(self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet)
        data_test = conllx_stacked_data.read_stacked_data_to_variable(self.test_path, self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.type_alphabet, use_gpu=False,
                                                                      prior_order=self.prior_order)
        for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test, 1, use_gpu=False):
            input_encoder, _, sentences = batch
            word, char, pos, heads, types, masks, lengths = input_encoder

            heads_pred, types_pred, _, _ = self.dependency.decode(word, char, pos, mask=masks, length=lengths, beam=1, leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()

            result = pred_writer.write_system(sentences, word, pos, heads_pred, types_pred, lengths, symbolic_root=True)

        return result

    def simplify(self, sentence):
        RESULT = self.oo.simplify_dep_parse(sentence)
        RESULT = self.oo.merge_linear_dep(RESULT, sentence)
        RESULT = self.oo.assign_impt_factor(RESULT, 1)
        result = self.oo.print_dep_parse(sentence, RESULT)
        return result

    def processing(self, sentence):
        conll = a.morphologyAnalysis(sentence)
        with open("tmp.txt", 'w') as f:
            f.write(conll)
        parsing = a.dependencyparsing()
        result = a.simplify(parsing)
        result.setdefault("conll", parsing)
        return result