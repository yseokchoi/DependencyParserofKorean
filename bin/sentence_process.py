import re

import transformer.Constants as Constants
from UPosTagMap import *
from SimplifyDepParse3 import SimplifyDepParse


class SentenceProcessing(object):
    def __init__(self, symbol_file="./models/symbols.txt", debug=False):
        self.debug = debug
        self._symbol2key = self.readSymbolFile(symbol_file)
        self._hanja = "([\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff]+)"
        self._english = "([a-zA-Z]+)"
        self._number = "([0-9]+)"
        self._sharp = "(#+)"
        self._key = "([A-Z]+##)"
        self._pattern_sharp = re.compile(self.sharp)
        self._pattern_english = re.compile(self.english)
        self._pattern_hanja = re.compile(self.hanja)
        self._pattern_number = re.compile(self.number)
        self._patterns = dict()
        for symbol in list(self.symbol2key.keys()):
            symbol_items = [re.escape(x) for x in self.symbol2key[symbol] if x != "#"]
            re_symbol_item = "|".join(symbol_items)
            self._patterns.setdefault(symbol, re.compile("({})".format(re_symbol_item)))

        self._key_pattern = re.compile(self._key)
        self._pattern_punctuations = re.compile("^[\.\?!,:;]+$")

        self.upostag_map = UPosTagMap()
        self.oo = SimplifyDepParse()

        self.unicode_symbol()


    @property
    def symbol2key(self):
        return self._symbol2key

    @property
    def hanja(self):
        return self._hanja

    @property
    def english(self):
        return self._english

    @property
    def number(self):
        return self._number

    @property
    def sharp(self):
        return self._sharp

    @property
    def pattern_sharp(self):
        return self._pattern_sharp

    @property
    def pattern_english(self):
        return self._pattern_english

    @property
    def pattern_hanja(self):
        return self._pattern_hanja

    @property
    def pattern_number(self):
        return self._pattern_number

    @property
    def patterns(self):
        return self._patterns

    @property
    def key_pattern(self):
        return self._key_pattern

    @property
    def pattern_punctuations(self):
        return self._pattern_punctuations

    def debug_print(self, sentence):
        print(sentence)

    def unicode_symbol(self):
        self.unicode_symbols = {
            "ᄀ": "ㄱ",
            "ᆫ": "ㄴ",
            "ᄃ": "ㄷ",
            "ᆯ": "ㄹ",
            "ᄆ": "ㅁ",
            "ᄇ": "ㅂ",
            "ᆼ": "ㅇ",
            "ᄎ": "ㅊ",
            "ᄒ": "ㅎ",
            "，": ",",
            "`": "\'",
            "\'": "\'",
            "\"": "\"",
            "-": "-",
            "‘": "\'",
            "“": "\"",
            "­": "-",
            "”": "\"",
            "─": "-",
            "―": "-",
            "－": "-",
            "_": "_",
            "’": "\'",
            "…": "...",
            "...": "...",
            "…………": "...",
            "∼": "~",
            "～": "~",
            "~": "~",
            "○": "O",
            "O": "O",
            "X": "X",
            "×": "x",
            "+": "+",
            ".": ".",
            "?": "?",
            "!": "!",
            "/": "/",
            ",": ",",
            "·": "·",
            ";": ";",
            "(": "(",
            ")": ")",
            "[": "[",
            "]": "]",
            "{": "{",
            "}": "}",
            "%": "%",
            "&": "&",
            "@": "@",
            "#": "#",
            "*": "*",
            "^": "^",
            "〓": "=",
            "<": "<",
            ">": ">"
        }

    def readSymbolFile(self, filepath):
        symbol2key = dict()
        file_contents = [x.strip() for x in open(filepath, "r").read().split("\n\n") if x != ""]

        for file_content in file_contents:
            for idx, line in enumerate(file_content.split("\n")):
                line = line.strip()
                if line == "":
                    continue

                if idx == 0:
                    key = line.split(":")[0]
                    symbol2key.setdefault(key, {})
                else:
                    entity, freq = line.split(": ")
                    symbol2key[key].setdefault(entity, int(freq))

        if self.debug:
            self.debug_print("Find the Symbol Part-of-Speech({})".format(len(symbol2key)))
            self.debug_print("Symbol Keys: {}".format(list(symbol2key.items())))

        return symbol2key

    def unicode_to_symbol(self, sentence):
        result = sentence

        for key1, key2 in self.unicode_symbols.items():
            key1_, key2_ = re.escape(key1), re.escape(key2)
            if re.search(key1_, result):
                result = re.sub(key1_, key2, result)
        return result

    def check_symbol(self, entity, key2entity):

        for m in re.finditer(self.pattern_sharp, entity):
            m_str = m.group(1)
            if self.debug:
                self.debug_print("({}): Find the Escape Sharp({})".format(entity, m_str))
            key2entity["SW##"].append(m_str)
        entity = self.pattern_sharp.sub("SW##", entity)

        if "SW##" not in entity:
            for m in re.finditer(self.pattern_english, entity):
                m_str = m.group(1)
                if self.debug:
                    self.debug_print("({}): Find the English({})".format(entity, m_str))
                key2entity["SL##"].append(m_str)
            entity = self.pattern_english.sub("SL##", entity)

        for m in re.finditer(self.pattern_hanja, entity):
            m_str = m.group(1)
            if self.debug:
                self.debug_print("({}): Find the Hanja({})".format(entity, m_str))
            key2entity["SH##"].append(m_str)
        entity = self.pattern_hanja.sub("SH##", entity)

        for m in re.finditer(self.pattern_number, entity):
            m_str = m.group(1)
            if self.debug:
                self.debug_print("({}): Find the Number({})".format(entity, m_str))
            key2entity["SN##"].append(m_str)
        entity = self.pattern_number.sub("SN##", entity)

        for symbol in list(self.symbol2key.keys()):
            pattern = self.patterns[symbol]
            for m in re.finditer(pattern, entity):
                m_str = m.group(1)
                if self.debug:
                    self.debug_print("({}): Find the Symbol({})".format(entity, m_str))
                key2entity["{}##".format(symbol)].append(m_str)
            entity = pattern.sub("{}##".format(symbol), entity)

        return entity

    # 190410 added
    def prefix_final_or_pause_punctuation(self, sentence):
        convert_sentence = []

        tokens = sentence.split(" ")
        for token in tokens:
            if self.pattern_punctuations.search(token):
                convert_sentence[-1] += token
            else:
                convert_sentence.append(token)

        return " ".join(convert_sentence)

    def divided_brackets(self, sentence):
        divided_tokens = []
        org_sentence = []
        org_sentence_space_info = []

        tokens = sentence.split(" ")
        for token in tokens:
            result_tmp_tokens = []
            checksymbol = False
            if "[" in token and "]" not in token:
                token = token.replace("[", " [ ")
                checksymbol = True

            if "[" not in token and "]" in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                if re.search("\][\.\?!]+$", token):
                    token = token.replace("]", " ]")
                else:
                    token = token.replace("]", " ] ")
                checksymbol = True

            if "(" in token and ")" not in token:
                token = token.replace("(", " ( ")
                checksymbol = True

            if "(" not in token and ")" in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                if re.search("\)[\.\?!]+$", token):
                    token = token.replace(")", " )")
                else:
                    token = token.replace(")", " ) ")
                checksymbol = True

            if "{" in token and "}" not in token:
                token = token.replace("{", " { ")
                checksymbol = True

            if "{" not in token and "}" in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                if re.search("\}[\.\?!]+$", token):
                    token = token.replace("}", " }")
                else:
                    token = token.replace("}", " } ")
                checksymbol = True

            if "「" in token and "」" not in token:
                token = token.replace("「", " 「 ")
                checksymbol = True

            if "「" not in token and "」" in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                if re.search("」[\.\?!]+$", token):
                    token = token.replace("」", " 」")
                else:
                    token = token.replace("」", " 」 ")
                checksymbol = True

            if "『" in token and "』" not in token:
                token = token.replace("『", " 『 ")
                checksymbol = True

            if "『" not in token and "』" in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                if re.search("』[\.\?!]+$", token):
                    token = token.replace("』", " 』")
                else:
                    token = token.replace("』", " 』 ")
                checksymbol = True

            if "<" in token and ">" not in token:
                token = token.replace("<", " < ")
                checksymbol = True

            if "<" not in token and ">" in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                if re.search(">[\.\?!]+$", token):
                    token = token.replace(">", " >")
                else:
                    token = token.replace(">", " > ")
                checksymbol = True

            if "《" in token and "》" not in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                token = token.replace("《", " 《 ")
                checksymbol = True

            if "《" not in token and "》" in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                if re.search("》[\.\?!]+$", token):
                    token = token.replace("》", " 》")
                else:
                    token = token.replace("》", " 》 ")
                checksymbol = True

            if "【" in token and "】" not in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                token = token.replace("【", " 【 ")
                checksymbol = True

            if "【" not in token and "】" in token:
                # 190412 닫는 괄호 후 마침표만 있을 경우에 띄어주지 않도록 하는 조건문 생성
                if re.search("】[\.\?!]+$", token):
                    token = token.replace("】", " 】")
                else:
                    token = token.replace("】", " 】 ")
                checksymbol = True

            if token.count("\'") == 1:
                token = token.replace("\'", " \' ")
                checksymbol = True

            if token.count("\"") == 1:
                token = token.replace("\"", " \" ")
                checksymbol = True

            if token.count("˝") == 1:
                token = token.replace("˝", " \" ")
                checksymbol = True

            if token.count(":") == 1:
                token = token.replace(":", " : ")
                checksymbol = True

            if token.count("-") == 1:
                token = token.replace("-", " - ")
                checksymbol = True

            if not checksymbol:
                result_tmp_tokens.append(token)
            else:
                tmp_tokens = [x for x in token.split(" ") if x != ""]
                result_tmp_tokens.extend(tmp_tokens)

            result_tmp_tokens_length = len(result_tmp_tokens)

            for idx, t in enumerate(result_tmp_tokens, start=1):
                divided_tokens.append(t)
                divided_tokens.append("<sa>")
                org_sentence.append(t)
                if idx == result_tmp_tokens_length:
                    org_sentence_space_info.append(False)
                else:
                    org_sentence_space_info.append(True)

        return divided_tokens[:-1], " ".join(org_sentence), org_sentence_space_info

    def getCharacter(self, entity):
        characters = []

        tmp_entity = entity
        while tmp_entity != "":
            tmp_regx = re.search("(^[A-Z]+##)", tmp_entity)
            if tmp_regx:
                symbol_character = tmp_regx.group()
                symbol_character_length = len(symbol_character)
                characters.append(symbol_character)
                tmp_entity = tmp_entity[symbol_character_length:]
            else:
                characters.append(tmp_entity[0])
                tmp_entity = tmp_entity[1:]
        return characters

    def input_conversion(self, sentence):
        sentence = " ".join([x.strip() for x in sentence.split(" ") if x.strip() != ""])
        sentence = self.prefix_final_or_pause_punctuation(sentence)
        sentence = self.unicode_to_symbol(sentence)
        symbol_mapping = {"SH##":[], "SL##":[], "SN##":[]}
        for symbol in list(self.symbol2key.keys()):
            symbol_mapping.setdefault("{}##".format(symbol), [])

        sentence_tokens, org_of_sentence, org_space_info_of_sentence = self.divided_brackets(sentence)
        input_sentence_characters = []
        for token in sentence_tokens:
            if token == "<sa>":
                input_sentence_characters.append(token)
            else:
                token = self.check_symbol(token, symbol_mapping)
                input_sentence_characters.extend(self.getCharacter(token))

        if self.debug:
            self.debug_print("ORG Sentence: {}\nConversion Sentence: {}\nMapping Symbols: {}\n\n".format(sentence, " ".join(input_sentence_characters), symbol_mapping.items()))
        return " ".join(input_sentence_characters), symbol_mapping, org_of_sentence, org_space_info_of_sentence

    def input_conversion_sentences(self, sentences):
        """
        :param sentences: [sentence_1, sentence_2, sentence_3,...]
        :return: [[character of sentence_1], [character of sentence_2], ...]], [[{SF##: [...], SN##: [...], ...}], ...]
        """
        org_sentences = []
        org_sentence_space_infos = []
        input_sentences = []
        sentence_symbol_mappings = []
        for sent in sentences:
            input_sentence, sentence_symbol_mapping, org_of_sentence, org_space_info_of_sentence = self.input_conversion(sent)
            input_sentences.append(input_sentence)
            sentence_symbol_mappings.append(sentence_symbol_mapping)
            org_sentences.append(org_of_sentence)
            org_sentence_space_infos.append(org_space_info_of_sentence)

        return org_sentences, org_sentence_space_infos, input_sentences, sentence_symbol_mappings

    def get_instances_for_morphology(self, sentences, max_sent_len, system=True):
        error_code = -1
        error_msg = ""

        word_insts = []
        trimmed_sent_count = 0
        for sent in sentences:
            words = sent.split()

            if len(words) > max_sent_len:
                trimmed_sent_count += 1
                if system:
                    error_code = 11
                    error_msg = "문장의 길이가 너무 길어 파싱을 할 수 없습니다."
                    return error_code, error_msg, []

            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

        return error_code, error_msg, word_insts

    def convert_instance_to_idx_seq_for_morphology(self, word_insts, word2idx, sentence_symbol_mappings):
        word_id_insts = []

        for s, symbol_mapping in zip(word_insts, sentence_symbol_mappings):
            symbol_mapping.setdefault("<unk>", [])
            id_inst = []
            for w in s:
                if w in word2idx:
                    id_inst.append(word2idx[w])

                else:
                    id_inst.append(Constants.UNK)
                    symbol_mapping["<unk>"].append(w)
            word_id_insts.append(id_inst)
        return word_id_insts, sentence_symbol_mappings

    def convert_key_to_symbol(self, sentences, sentence_symbol_mappings):
        convert_sentences = []

        for sentence, symbol_mapping in zip(sentences, sentence_symbol_mappings):
            index_dict = {x: 0 for x in list(symbol_mapping.keys())}

            convert_sentence = []
            tokens = sentence.split(" ")
            for token in tokens:
                if token == "<unk>" and index_dict["<unk>"] < len(symbol_mapping["<unk>"]):
                    convert_sentence.append(symbol_mapping["<unk>"][index_dict["<unk>"]])
                    index_dict["<unk>"] += 1

                elif token == "<unk>":
                    continue

                elif token == "<s>":
                    continue

                elif self.key_pattern.search(token):
                    key_ = self.key_pattern.search(token).group(1)
                    if index_dict[key_] < len(symbol_mapping[key_]):
                        convert_sentence.append(symbol_mapping[key_][index_dict[key_]])
                        index_dict[key_] += 1
                else:
                    convert_sentence.append(token)

            if self.debug:
                self.debug_print("Function: convert_key_to_symbol")
                self.debug_print("Sentence: {}\nSymbol Mappings: {}\nConvert Sentence: {}\n\n".format(sentence, symbol_mapping, " ".join(convert_sentence)))

            convert_sentences.append(" ".join(convert_sentence))
        return convert_sentences

    def character_to_morphology(self, input_sentences, output_sentences, system=False):
        convert_sentences = []
        no_convert_sentences = []
        error_code = -1
        error_msg = ""

        for inp, out in zip(input_sentences, output_sentences):
            if self.debug:
                self.debug_print("Function: character_to_morphology")
                self.debug_print("Input Sentence({}): {}".format(len([x.strip() for x in inp.split("<sa>") if x.strip() != ""]), [x.strip() for x in inp.split("<sa>") if x.strip() != ""]))
                self.debug_print("Output Sentence({}): {}\n\n".format(len([x.strip() for x in out.split("<sa>") if x.strip() != ""]), [x.strip() for x in out.split("<sa>") if x.strip() != ""]))

            if len([x.strip() for x in inp.split("<sa>") if x.strip() != ""]) != len([x.strip() for x in out.split("<sa>") if x.strip() != ""]):
                no_convert_sentences.append((inp, out))
                convert_sentences.append(None)
                continue

            eojuls = [x.strip() for x in out.split("<sa>") if x.strip() != ""]
            convert_sentence = []

            for idx, eojul in enumerate(eojuls, start=1):
                tokens = eojul.split(" ")
                m_tmp = []
                for token in tokens:
                    if re.search("<[A-Z]+>", token):
                        if len(m_tmp) > 0:
                            morph = "".join(m_tmp)
                            pos = token.replace("<", "").replace(">", "")
                            tok = "{}/{}".format(morph, pos)
                            convert_sentence.append(tok)
                        m_tmp = []
                    else:
                        m_tmp.append(token)
                convert_sentence.append("<sa>")
            convert_sentences.append(" ".join(convert_sentence[:-1]))

        # convert_sentences = self.prefix_final_punctuation(convert_sentences)  # 180410 final punctuation 처리

        if system and len(no_convert_sentences) > 0:
            error_code = 12
            error_msg = "형태소 분석을 수행할 수 없습니다.(Error Code: {})".format(error_code)
            return error_code, error_msg, convert_sentences, no_convert_sentences

        return error_code, error_msg, convert_sentences, no_convert_sentences

    def convert_to_CoNLL(self, real_sentences, org_sentences, org_sentence_space_infos, output_sentences):
        conll_sentences = []

        for real_sentence, org_sentence, org_space_info, output_sentence in zip(real_sentences, org_sentences, org_sentence_space_infos, output_sentences):
            if output_sentence is None:
                continue
            sentence = ["#SENT: {}".format(real_sentence)]
            org_tokens = [x.strip() for x in org_sentence.split(" ")]
            out_tokens = [x.strip() for x in output_sentence.split("<sa>") if x.strip() != ""]

            for idx, (token, org_token, space_info) in enumerate(zip(out_tokens, org_tokens, org_space_info), start=1):
                only_morph = []
                only_pos = []

                for mo in token.split(" "):
                    if mo == "//SP":
                        m, p = "/", "SP"
                    else:
                        m, p = mo.split("/")

                    only_morph.append(m)
                    only_pos.append(p)
                input_xpostag = " ".join(only_pos)
                lemma = " ".join(only_morph)
                if space_info:
                    tmp = [str(idx), org_token, " ".join(only_morph), "+".join(self.upostag_map.GetUPOS(input_xpostag, lemma)),
                           "+".join(only_pos), "_", "0", "root", "_", "SpaceAfter=No"]
                else:
                    tmp = [str(idx), org_token, " ".join(only_morph), "+".join(self.upostag_map.GetUPOS(input_xpostag, lemma)),
                           "+".join(only_pos), "_", "0", "root", "_", "_"]
                sentence.append("\t".join(tmp))
            conll_sentences.append(sentence)

        return conll_sentences

    def simplify(self, sentence):
        RESULT = self.oo.simplify_dep_parse(sentence)
        RESULT = self.oo.merge_linear_dep(RESULT)
        RESULT = self.oo.assign_impt_factor(RESULT, 1)
        result = self.oo.print_dep_parse(sentence, RESULT)
        return result









