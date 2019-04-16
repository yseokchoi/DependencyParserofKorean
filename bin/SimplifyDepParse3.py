# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:22:59 2018

@author: ADMIN
"""

class SimplifyDepParse():
    
    
    
    def __init__(self, CR=1):
        
        # compression rate: 1~depth_tree
        self.compression_rate = CR
        self.max_cr = 10
        
        self.HEAD_ID = 6
        self.REL_ID = 7
        self.SpaceAfter = 9
        
        self.NP_UPOS = ('NOUN', 'PRON', 'PROPN', 'NUM', 'X') #, 'SYM')
        self.VP_UPOS = ('VERB', 'ADJ', 'AUX')        
        
        self.NPOS = ('NNG', 'NNP', 'NNB', 'NR', 'NP', 'XR')
        self.JPOS = ('JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC') 
        
        #self.OutFile = open('TMP_OUT.txt', 'w', encoding='utf8')
        self.printable = 1
        
        self.display_node_unit = 12
        self.min_display_nodes = 5
        self.max_display_nodes = 18
        self.max_tree_depth = 99
        
        


    def merge_node(self, RESULT, head, child_lst):
        
        for child in child_lst:
            RESULT[head][0] += RESULT[child][0] + [child]
        
            if child in RESULT[head][3]:
                RESULT[head][2] -= 1
                RESULT[head][3].remove(child)        
            
        tmp = child_lst + [head]
        sorted(tmp)
        RESULT[head][4] = RESULT[tmp[-1]][4]
        
        for child in child_lst:        
            del RESULT[child]
    
    def check_linear_cond_1(self, SENT, RESULT, w_id): # 빛_같은_것
        
        morphs = SENT[w_id][2].split(' ')
        if morphs[0] != '것' : 
            return (None, None)
        
        [adjunct_ids, head_info, child_count, child_ids, new_rel, chunks, CR] = RESULT[w_id]
        
        child_id = child_ids[0]
        if RESULT[child_id][2] == 1:
            g_child_id = RESULT[child_id][3][0]
            if w_id - 1 == child_id and child_id -1 == g_child_id:
                if SENT[child_id][1] == '같은':
                    return (g_child_id, [w_id, child_id]) 
         
        return (None, None)
    
    def check_linear_cond_2(self, SENT, RESULT, w_id, child_id):
        
        if RESULT[child_id][0]:
            tmp = [child_id] + RESULT[child_id][0]
            child_id = sorted(tmp)[-1]
            
        if child_id + 1 != w_id: return False
        morphs = SENT[w_id][2].split(' ')
        pos = SENT[w_id][4].split('+')
        child_pos = SENT[child_id][4].split('+')
        if pos[0] == 'NNB' and child_pos[-1] == 'ETM' and morphs[0] in ('지', '듯', '듯이', '양', '대로', '것', '데', '바'): # 닫힌_지  질식할_듯
            return True
        
        if child_pos[-1] == 'ETM' and morphs[0] in ('이전', '이후', '때', '후', '뒤', ):
            return True
      
        return False
    

    
    def check_linear_cond_3_4_5(self, SENT, RESULT, w_id, child_id):
        

        chunks = [child_id]
        if RESULT[child_id][0] != []:
            chunks += RESULT[child_id][0]
            chunks = sorted(chunks)        
            
        child_id = chunks[-1]
        if child_id + 1 != w_id: return False
        
        lex, morphs = SENT[w_id][1], SENT[w_id][2].split(' ')
        pos = SENT[w_id][4].split('+')
        child_pos, child_posLst = SENT[child_id][4], SENT[child_id][4].split('+') 

        # [ 너_따위가 ]  [ 얼음_대신에 ]  
        # [ 앙드레김 씨가 ]
        if morphs[0] in ('따위','대신', '씨'):
            if child_pos[0] in self.NPOS and child_pos[-1] not in self.JPOS  : 
                return True

        # [ 어렵기 때문에 ]  [ 보지_못했기 때문에 ]
        # [ 동생 때문에 ]
        # 스스로 [떳떳치 못했기 때문이] 아닌가 여겨진다        
        elif lex in ('때문에', '때문이',):
            if child_posLst[-1] == 'ETN':
                return True            
            if child_posLst[-1] in self.NPOS or child_posLst[-1] in ('XSN', ):
                return True 

        # [ 건강을 위해 ]  
        # 아이들을 [ 치료하기 위해 ] 사용하는
        elif lex.find('위해') == 0 and morphs[0] == '위하' and pos[0] == 'VV':
            if child_posLst[-1] in ('JKO', 'ETN'):
                return True
            
        # [선언을 통해 ]
        elif lex.find('통해') == 0 and morphs[0] == '통하' and pos[0] == 'VV':
            if child_posLst[-1] == 'JKO':
                return True   
        
        # [ 스웨덴을 비롯, ]  [ 스웨덴을 비롯해 ]  [스웨덴을 비롯하여 ]
        elif lex.find('비롯') == 0 and morphs[0] == '비롯':
            if child_posLst[-1] == 'JKO':
                return True         
        
        # [ 법률에 의해 ]  [ 관계에 의하여 ]
        elif (lex.find('의해') == 0 or morphs[0] == '의하') and pos[0] == 'VV':
            if child_posLst[-1] == 'JKB':
                return True
            
        # [ 부인병 예방에 관한 ]  [ 김정일에 관해서는 ]  [ 이번 사건에 관해 ]
        elif (lex.find('관해') == 0 or morphs[0] == '관하') and pos[0] == 'VV':
            if child_posLst[-1] == 'JKB':
                return True            
        
        # [ 참가에 따라 ] 
        elif lex.find('따라') == 0 and morphs[0] == '따르' and pos[0] == 'VV':
            if child_posLst[-1] == 'JKB':
                return True
            
      
        return False     
    
    def check_linear_cond_AUX(self, SENT, RESULT, w_id):
        
        aux_expr = []
        while True:
            if w_id == 0: break
        
            head_id = RESULT[w_id][1][0][0]
            if head_id == w_id-1 and SENT[w_id][3] not in ('PUNCT', ):
                aux_expr.append(w_id)
                w_id = head_id
            elif SENT[w_id][3] in ('VERB', 'ADJ', 'AUX'): # AUX는 사실.. 오류임
                aux_expr.append(w_id)
                break
            else:
                break
            
        if len(aux_expr) <= 1 : return [] 
        aux_expr = sorted(aux_expr) 
        
        if SENT[aux_expr[0]][3] not in ('VERB', 'ADJ', 'AUX'): return []
            
        return aux_expr            
    
    def check_linear_cond_pseudo_AUX(self, SENT, RESULT, w_id, child_id):
        
        # 움직이기도 하는_모양이다
        # 가고_있음이니...
        # 본래는 AUX로 인식되어야 하는 경우임.
        chunks = [child_id]
        if RESULT[child_id][0] != []:
            chunks += RESULT[child_id][0]
            chunks = sorted(chunks)
        child_id = chunks[-1]            
            
        if child_id + 1 != w_id: return False
        lex, morphs = SENT[w_id][1], SENT[w_id][2].split(' ')
        UPOS, pos = SENT[w_id][3], SENT[w_id][4].split('+')
        childUPOS, child_pos, child_posLst = SENT[child_id][3], SENT[child_id][4], SENT[child_id][4].split('+') 
        
        if childUPOS in ('VERB', 'ADJ') and UPOS == 'AUX' and pos[0] == 'VX':
            return True
                
        # [ 만들었다고 한다 ]
        if morphs[0] == '하' and pos[0] in ('VV', 'VX') and child_pos in ('VV+ETN+JX', 'VA+ETN+JX', 'VV+EP+EC'):            
            return True
        
        # [쥐고 있는 듯싶다 ] [ 움직일 성싶다 ]
        if len(pos) > 1 and morphs[0] in ('듯', '성',) and pos[0] == 'NNB'  and pos[1] == 'VX' and child_posLst[-1] == 'ETM':
            return True        
        
      
        return False  
    
    def check_linear_cond_num1(self, SENT, RESULT, w_id, head_id):
        
        # 19만 3천 6백원을   
        # TODO: [ 20/SN+여/XSN 곳에서 ] 
        # TODO: [ 10명 미만의 ]
        num_expr = []
        while True:
            
            pos = SENT[w_id][4].split('+')
            if pos[0] in ('SN', 'NR') :
                num_expr.append(w_id)
            elif pos[0] == 'NNB':
                num_expr.append(w_id)
                break
            else:
                break
            if pos[-1] not in ('SN', 'NR'):
                break
            w_id += 1
            if w_id == self.sent_len: break
           
        if len(num_expr) <= 1 : return []
        num_expr = sorted(num_expr)  
        for n in num_expr[:-1]:
            if RESULT[n][1][0][0] not in num_expr: return []
            
        return num_expr
    
    def check_linear_cond_num2(self, SENT, RESULT, w_id, head_id):
        
        # [ 스물/NR 한/MM 과목/NNG ]  [ 서른/NR 다섯/MM 살/NNB ]  [ 스물/NR 일곱/NR 의/JKG ]  
        # [ 스물두/NR+MM 살/NNB ]                     
        num_expr = []
        while True:
            
            pos, posLst = SENT[w_id][4], SENT[w_id][4].split('+')
            if pos in ('NR', 'MM', 'NR+MM', 'NR+NR') :
                num_expr.append(w_id)
            elif posLst[0] in ('NR', 'MM') and posLst[-1] not in ('NR', 'MM'):
                num_expr.append(w_id)
                break
            elif posLst[0] in ('NNB', 'NNG'):
                num_expr.append(w_id)
                break            
            else:
                break
            w_id += 1
            if w_id == self.sent_len: break
            
        if len(num_expr) <= 1 : return []
        num_expr = sorted(num_expr)
        for n in num_expr[:-1]:
            if RESULT[n][1][0][0] not in num_expr: return []        
        return num_expr
    
    def check_linear_cond_date_time(self, SENT, RESULT, w_id, head_id):
        
        # 86년 10월 20일까지 
        # 11일 오전 9시                       
        # 오후 5시부터
        num_expr = []
        
        time_start_ind = False
        date = False
        time = False
        begin_w_id = w_id
        
        while True:
            
            lex = SENT[w_id][1]
            pos = SENT[w_id][4].split('+')
            morphs = SENT[w_id][2].split(' ')
            
            if lex in ('오전', '오후', '새벽', '저녁', '낮'):
                time_start_ind = True
                num_expr.append(w_id)
            elif pos[0] in ('SN', 'NR') and len(morphs) >= 2 and morphs[1] in ('년','월','일') and time_start_ind == False:
                num_expr.append(w_id)
                date = True
            elif pos[0] in ('SN', 'NR') and len(morphs) >= 2 and morphs[1] in ('시', '분', '초'):
                num_expr.append(w_id)  
                time = True
            else:
                break
            
            if morphs[-1] not in ('년','월','일', '시', '분', '초', '오전', '오후', '새벽', '저녁', '낮'):
                break
            
            w_id += 1
            if w_id == self.sent_len: break
            

        if time_start_ind == False and time and begin_w_id -1 >= 0 and SENT[begin_w_id-1][1] in ('오전', '오후', '새벽', '저녁', '낮'):
            num_expr.append(begin_w_id-1)
            
        if len(num_expr) <= 1 : return []            
        num_expr = sorted(num_expr)
        for n in num_expr[:-1]:
            if RESULT[n][1][0][0] not in num_expr: return []        
        return num_expr
        
    def check_adv_without(self, SENT, RESULT, w_id):
        
        # [ 어쩔 수 없이 ]  [ 하는 수 없이 ] [ 자신의 반성 없이 ]
        adv_expr = []
        while True:
            adv_expr.append(w_id)
            if RESULT[w_id][2] == 1:
                c_id = RESULT[w_id][3][0]
            elif RESULT[w_id][2] == 0:
                break
            else:
                return []
            
            w_id = c_id
            if w_id == 0: break
            
        if len(adv_expr) == 1: return []
        return(sorted(adv_expr))
    
    def check_compound_noun(self, head_UPOS, head_xpos, head_xposLst, head_lex, head_morphs):
        
        if 'NNB' in head_xposLst: return False
        if 'SN' in head_xposLst: return False
        if head_morphs[0] in ('대신', ): return False
        #if len(head_morphs[0]) == 1: return False   # WHY?  [신라_때] [전립선_암]
                        
        return True

                  
    def init_RESULT(self, SENT):
        
        RESULT = dict()        
        count_child = [[0, []]] + [[0, []] for _ in SENT]
        for w in SENT:
            w_id = int(w[0])
            head_id = int(w[self.HEAD_ID])
            rel = w[self.REL_ID]

            value = [[], [[head_id, rel]], 0, [], None, [], self.compression_rate]
            RESULT[w_id] = value
            count_child[head_id][0] += 1
            count_child[head_id][1].append(w_id)
        self.root_id = count_child[0][1][0]
        for idx in range(1, len(count_child)):
            RESULT[idx][2] = count_child[idx][0]
            RESULT[idx][3] = count_child[idx][1]
        
        SENT.insert(0, [0, None]) 
        self.sent_len = len(SENT)
    
        return RESULT        
        
    def change_child(self, RESULT, w_id, remove_child, add_child):
        if w_id == 0:
            self.root_id = add_child
            return
        RESULT[w_id][3].remove(remove_child)
        RESULT[w_id][3].append(add_child)
        
    def assign_imp_factor_rec(self, cur_id, cr, RESULT):
        
        rel = RESULT[cur_id][1][0][1]
        
        '''if cr != self.max_cr:
            if rel not in ('nsubj', 'obj', 'root'): 
                cr += 1
            if rel in ('punct', 'det',):
                cr = self.max_cr'''
          
        cr += 1
        RESULT[cur_id][6] = cr
        
        if RESULT[cur_id][2] == 0: return               
        
        for c_id in RESULT[cur_id][3]:
            self.assign_imp_factor_rec(c_id, cr, RESULT)
            
        return
         
    def assign_impt_factor(self, RESULT, cr):
        
        # RESULT        
        #               0               1                  2            3         4             5              6
        # w_ID:[ [adjunct_ids], [[head_ID, rel],...], child_count, [child_ids], NewRel, [chunk_kind,..], CompressionRate ]        
        #   
        w_id = 0
        while w_id < self.sent_len:
            w_id += 1
            
            if w_id not in RESULT:
                continue
            
            if RESULT[w_id][1][0][0] == 0:
                head_id = w_id
                break
            
        self.assign_imp_factor_rec(head_id, 0, RESULT)

            
        return RESULT     
    
    def merge_linear_dep(self, RESULT):
        
        results = sorted([k for k in RESULT.keys()])
        len_results = len(results)
        for x_id in range(0, len_results):
            
            w_id = results[x_id]
            if w_id not in RESULT: continue
            [adjunct_ids, head_info, child_count, child_ids, rel_id, chunks, CR] = RESULT[w_id]
            # UPOS = SENT[w_id][3]
            # xpos, xposLst = SENT[w_id][4], SENT[w_id][4].split('+')
            
            if child_count == 0 and adjunct_ids == []:
                child_id = w_id
                head_id = head_info[0][0]
                tmp_child_lst = [child_id]
                
                while head_id > 0 and head_id in RESULT:
                    if RESULT[head_id][2] > 1: break
                    if RESULT[head_id][0] != []: break
                    tmp_child_lst += [head_id]
                    head_id = RESULT[head_id][1][0][0]
                if len(tmp_child_lst) > 1:
                    self.merge_node(RESULT, tmp_child_lst[-1], tmp_child_lst[:-1])
                    RESULT[tmp_child_lst[-1]][5].append('linear')
        return RESULT
                    
                    
    def simplify_dep_parse(self, SENT):
        
        # RESULT        
        #               0               1                  2            3         4             5              6
        # w_ID:[ [adjunct_ids], [[head_ID, rel],...], child_count, [child_ids], NewRel, [chunk_kind,..], CompressionRate ]        
        #
        RESULT = self.init_RESULT(SENT)
        
        #### forward direction
        # TO_DO:
        # 책 몇 권이
        # 움직일 뿐, 
        # 여성들_역시
        # [ 약 4천만 원을 ]
        # [잠시 뒤]
        w_id = 0
        while w_id < self.sent_len:
            w_id += 1
            
            if w_id not in RESULT:
                continue
            
            [adjunct_ids, head_info, child_count, child_ids, rel_id, chunks, CR] = RESULT[w_id]
            
            #-----------------
            head_id, rel = head_info[0][0], head_info[0][1]
            
            if child_count >= 1:
                child_id = child_ids[0]
            
            UPOS = SENT[w_id][3]
            xpos, xposLst = SENT[w_id][4], SENT[w_id][4].split('+') 
            lex, morphs = SENT[w_id][1], SENT[w_id][2].split(' ')
            
            if head_id != 0:
                head_UPOS = SENT[head_id][3]
                head_xpos, head_xposLst = SENT[head_id][4], SENT[head_id][4].split('+')
                head_lex, head_morphs = SENT[head_id][1], SENT[head_id][2].split(' ')
            #-----------------                
                
            if child_count < 1 and (UPOS == 'NUM' or xposLst[0] in ('SN', 'NR')):
            
                if xpos in ('NR', 'NR+MM', 'NR+NR', 'MM'):  # 스물두 개  
                    num_expr = self.check_linear_cond_num2(SENT, RESULT, w_id, head_id)
                    if num_expr:
                        self.merge_node(RESULT, num_expr[-1], num_expr[:-1])
                        RESULT[num_expr[-1]][5].append('QUANT')
                        w_id = num_expr[-1]
                        continue
                
                if xposLst[0] == 'SN':
                    num_expr = self.check_linear_cond_num1(SENT, RESULT, w_id, head_id)
                    if num_expr:
                        self.merge_node(RESULT, num_expr[-1], num_expr[:-1])
                        RESULT[num_expr[-1]][5].append('QUANT')  
                        w_id = num_expr[-1]
                        continue
                
                    else:
                        num_expr = self.check_linear_cond_date_time(SENT, RESULT, w_id, head_id)  
                        if num_expr:
                            self.merge_node(RESULT, num_expr[-1], num_expr[:-1]) 
                            RESULT[num_expr[-1]][5].append('DATE_TIME')
                            w_id = num_expr[-1]
                            continue
                  
                        
            if child_count == 0 and head_id == w_id + 1:  # [ w_id, head_id ]
                # linear structure
                                 
                if xpos == 'MAG' and lex in ('안', '못') and head_UPOS in ('VERB', 'ADJ'):   # [못 움직이다]
                    self.merge_node(RESULT, head_id, [w_id])
                    RESULT[head_id][5].append('NEG')
                    continue

                # Fixed Pattern: [다음과 같이]                
                if lex in ('다음과', '이와', '위와', '이상과', '전과') and head_lex in ('같이', ):   
                    self.merge_node(RESULT, head_id, [w_id])
                    continue                
                                        
                if UPOS == 'DET' and xpos == 'MM': # 이_세상에
                    self.merge_node(RESULT, head_id, [w_id])
                    continue
                   
                # [ X의  NOUN ]  [ 우리의 전통의 효사상을 ]                   
                if xposLst[-1] == 'JKG' and head_xposLst[0] in ('NNG', 'NNP'): 
                    self.merge_node(RESULT, head_id, [w_id])
                    RESULT[head_id][5].append('COMPOUND')
                    continue

                # [ 행정편의적 발상으로 ]                    
                if xposLst[0] == 'NNG' and xposLst[-1] == 'XSN' and morphs[-1] == '적' and head_xposLst[0] in ('NNG', 'NNP'): 
                    self.merge_node(RESULT, head_id, [w_id])
                    RESULT[head_id][5].append('COMPOUND')
                    continue                    
                    
                if xposLst[-1] in ('NNP', 'NNG') and head_xposLst[0] in ('NNG', 'NNP'): # 동경_올림픽
                    if self.check_compound_noun(head_UPOS, head_xpos, head_xposLst, head_lex, head_morphs):
                        self.merge_node(RESULT, head_id, [w_id])
                        RESULT[head_id][5].append('COMPOUND')
                        continue
                
                # expand...
                if xposLst[-1] == 'JKO' and head_morphs[0] == '하':  # 운동을_하
                    self.merge_node(RESULT, head_id, [w_id])
                    RESULT[head_id][5].append('NounPredicate')
                    continue
     
             
            if child_count == 1:   # [ child_id, w_id ]

                # 하는_수_없이  어쩔_수_없이
                if morphs[0] == '없이':  
                    adv_without_expr = self.check_adv_without(SENT, RESULT, w_id)
                    if adv_without_expr:
                        self.merge_node(RESULT, adv_without_expr[-1], adv_without_expr[:-1])
                        continue                                               
                                
                # 닫힌_지  질식할_듯이  숨긴_이후로  있는_것으로
                if self.check_linear_cond_2(SENT, RESULT, w_id, child_id): 
                    self.change_child(RESULT, head_id, w_id, child_id)              
                    self.merge_node(RESULT, child_id, [w_id])
                    RESULT[child_id][1][0][0], RESULT[child_id][1][0][1] = head_id, rel
                    RESULT[child_id][4] = w_id
                    continue

                # [동남아국가 등]      
                if morphs[0] == '등' and xposLst[0] == 'NNB':
                    self.change_child(RESULT, head_id, w_id, child_id)              
                    self.merge_node(RESULT, child_id, [w_id])
                    RESULT[child_id][1][0][0], RESULT[child_id][1][0][1] = head_id, rel
                    RESULT[child_id][4] = w_id
                    continue 
                    
                
                # 3, 4, 5가 모두 동일 조건임
                # HEAD 앞쪽 --> child가 child가 있어도 무방함
                # [ 학생 땨위가 ]  [ 얼음 대신 ]
                # [ 건강을_위해 ]  [ 참가에_따라 ] [ 선언을_통해 ] [ 스웨덴을_비롯 ]
                # 댄스가 생각보다 [어렵기 때문에]   
                # [ 동생 때문에 ]
                # 스스로 [떳떳치 못했기 때문이] 아닌가 여겨진다                
                if self.check_linear_cond_3_4_5(SENT, RESULT, w_id, child_id):
                    self.change_child(RESULT, head_id, w_id, child_id)              
                    self.merge_node(RESULT, child_id, [w_id])
                    RESULT[child_id][1][0][0], RESULT[child_id][1][0][1] = head_id, rel
                    RESULT[child_id][4]= w_id   # relation
                    continue
                
                    
                # AUX에서 빠진 경우...
                # [ 만들었다고 한다 ] [ 먹기도 하다 ]
                if self.check_linear_cond_pseudo_AUX(SENT, RESULT, w_id, child_id): 
                    RESULT[child_id][1] = RESULT[w_id][1]
                    self.change_child(RESULT, head_id, w_id, child_id)              
                    self.merge_node(RESULT, child_id, [w_id])                    
                    RESULT[child_id][5].append('PRED_AUX')
                    continue
                
                # [ "...의미" 라고 ]
                if UPOS == 'ADJ' and xposLst[0] == 'VCP': 
                    self.change_child(RESULT, head_id, w_id, child_id)              
                    self.merge_node(RESULT, child_id, [w_id])
                    RESULT[child_id][1][0][0], RESULT[child_id][1][0][1] = head_id, rel
                    RESULT[child_id][4] = w_id
                    continue                 
                        
                core_id, args = self.check_linear_cond_1(SENT, RESULT, w_id)  # 빛_같은_것
                if core_id:
                    RESULT[core_id][1] = RESULT[w_id][1]   # head-rel
                    self.change_child(RESULT, head_id, w_id, core_id)              
                    self.merge_node(RESULT, core_id, args)
                    continue
                    
                    
            # Predicate AUXilary
            if child_count == 0 and head_id == w_id-1 and head_id != 0:
                aux_expr = self.check_linear_cond_AUX(SENT, RESULT, w_id)
                if aux_expr:
                    self.merge_node(RESULT, aux_expr[0], aux_expr[1:]) 
                    RESULT[aux_expr[0]][5].append('PRED_AUX')
                    continue
                
            # punctuation, case... (non-linear)                
            if head_id != 0 and child_count == 0 and UPOS in ('PUNCT', 'PART', 'ADP') and xpos not in ('XPN',):
                self.merge_node(RESULT, head_id, [w_id])
                if UPOS in ('PART', 'ADP',):
                    RESULT[head_id][4]= w_id   # relation
                if lex in ('~', ):
                    RESULT[head_id][5].append('RANGE') 
                elif UPOS in ('PUNCT',) and 'QUOT' not in RESULT[head_id][5]:
                    RESULT[head_id][5].append('QUOT') 
                    
                continue
                
                    
            
        ############## 2nd LOOP
        results = sorted([k for k in RESULT.keys()])
        len_results = len(results)
        for x_id in range(0, len_results):
            
            w_id = results[x_id]
            if w_id not in RESULT: continue
            [adjunct_ids, head_info, child_count, child_ids, rel_id, chunks, CR] = RESULT[w_id]
            UPOS = SENT[w_id][3]
            xpos, xposLst = SENT[w_id][4], SENT[w_id][4].split('+')            
            
            if chunks == []: continue
        
            # 너무 짧은 인용은 다 포함하기...
            if 'QUOT' in chunks and child_count in (1, 2):
                adjunct_ids_sorted = sorted(adjunct_ids)
                valid_child = []
                merge_possible = True
                for c in child_ids:
                    if c > adjunct_ids_sorted[0] and c < adjunct_ids_sorted[-1]:
                        if RESULT[c][2] == 0 and 'QUOT' not in RESULT[c][5]:
                            valid_child.append(c)
                        else:
                            merge_possible = False
                            break
                if merge_possible and len(valid_child) <= 2:
                    self.merge_node(RESULT, w_id, valid_child)
                  
            # [ [3천_5백]  [~_6천_5백원] ]
            if 'RANGE' in chunks:
                if 'QUANT' in chunks or UPOS == 'NUM' or 'NR' in xposLst:
                    first_adjunct_id = sorted(adjunct_ids)[0]
                    quant_child = None
                    for c in child_ids:
                        if c == first_adjunct_id-1 or c in RESULT and first_adjunct_id-1 in RESULT[c][0]:
                            quant_child = c
                            break
                    if quant_child and RESULT[quant_child][2] == 0:
                        if 'QUANT' in RESULT[quant_child][5]:
                            self.merge_node(RESULT, w_id, [quant_child])
                        elif SENT[quant_child][3] == 'NUM' or 'NR' in SENT[quant_child][4].split('+'):
                            self.merge_node(RESULT, w_id, [quant_child])                   
                        
                       
            
        return RESULT    
        
    ########################################################    
    def node_lex_str(self, head_id, node_ids, ORG_SENT):
        
        w_str = [ORG_SENT[x][1]  for x in node_ids]
        
        w_str = []

        for x, y in zip(node_ids, node_ids[1:]+[0]):
            w = ORG_SENT[x][1]
            if x == head_id:
                w_str.append('['+w+']')
            else:
                w_str.append(w)                
                
        
        return '_'.join(w_str)
    
    def display_depths(self, SENT):
        
        tmp = sorted([[k, v] for k, v in SENT.items()])
        # 6: depth
        depth_lst = [int(x[6]) for (_, x) in tmp]
        max_depth = max(depth_lst)
        tmp = [0 for _ in range(0, max_depth+1)]
        for x in depth_lst:
            tmp[x] += 1
          
        num_nodes = len(SENT)
        
        disp_lst = []
        qq = 0
        idx = 1
        while idx < len(tmp):
            
            if qq + tmp[idx] >= self.display_node_unit:   
                if qq + tmp[idx] <= self.max_display_nodes:
                    disp_lst.append(idx)
                    qq = 0
                    idx += 1
                   
                elif tmp[idx] > self.max_display_nodes:
                    disp_lst.append(idx)
                    qq = 0
                    idx += 1
                
                else:
                    disp_lst.append(idx-1)
                    qq = 0
                   
                if disp_lst != [] and disp_lst[-1]+1 < len(tmp):
                    rest_nodes = sum(tmp[disp_lst[-1]+1:])
                    if rest_nodes < self.min_display_nodes:
                        disp_lst[-1] = max_depth
                        break
                
            else:
                qq += tmp[idx]
                idx += 1

              
        if disp_lst == [] or disp_lst[-1] != max_depth:
            disp_lst.append(max_depth)
            
            
        # temporary...  
        
        self.min_display_nodes
        tmp2 = [0]
        for z in range(1, len(tmp)):
            tmp2.append(tmp[z-1] + tmp[z])
          
        idx = 0
        qq = 0
        tmp3 = []
        for z in range(1, len(tmp)):
            qq += tmp[z]
            if z == disp_lst[idx]:
                tmp3.append(qq)
                qq = 0
                idx += 1
                
        #print(disp_lst, tmp3)
        return disp_lst, tmp3
 
            
    
    def print_dep_parse(self, ORG_SENT, SENT):
        OutFile = dict()
        # self.OutFile.write('\n=========================\n')
        #OutFile.append(str(len(ORG_SENT)-1) + '    ' + str(len(SENT)) + '\n')
        #OutFile.append('TOTAL node: ' +  str(len(SENT)) + '\n')
        OutFile.setdefault("total_node", len(SENT))
        
        disp_depths, ZZZ = self.display_depths(SENT)
        disp_depths_str = [str(x) for x in disp_depths]
        ZZZ_str = [str(x) for x in ZZZ]
        #OutFile.append('DISPLAY depths('+str(len(disp_depths))+'):  [' +  ' '.join(disp_depths_str) + '], [' + ' '.join(ZZZ_str) + ']\n')
        OutFile.setdefault("display_depths", len(disp_depths))
        
        if self.printable:
            print('----------\nROOT', self.root_id, len(ORG_SENT)-1, len(SENT))
        outtmp = []
        tmp = sorted([[k, v] for k, v in SENT.items()])
        for w in tmp:
            
            head_id = w[0]
            w_ids = sorted([w[0]] + w[1][0])
            
            if w[1][4]:
                w_new_rel = ORG_SENT[w[1][4]][1]
                #w_ids.remove(w[1][4])
            else:
                w_new_rel = "_"
                
            
            w_str = self.node_lex_str(head_id, w_ids, ORG_SENT)
            w_head = w[1][1][0][0]
            w_rel = w[1][1][0][1]

            if w[1][5] != []:                
                w_chunk = ','.join(w[1][5])
            else:
                w_chunk = "_"
                
            w_CR = str(w[1][6])
            if self.printable:
                print(w_CR , '\t', w[0], '_'.join(w_str), w_ids, '\t<--',  '('+str(w_head) + ', ' + w_rel +')', '    ['+ w_chunk +':'+ w_new_rel +']')
                
            
            res = [str(w_CR), str(w[0]), ' '.join([str(w) for w in w_ids]), w_str, str(w_head) + ':' + w_rel,  w_chunk + ':' + w_new_rel]
            outtmp.append(res)
        OutFile.setdefault("simplify", outtmp)
                    
        return OutFile
        
if __name__ == '__main__':

    FILE = '180816_result-non_head_final.txt'
    #FILE = 'sample.txt'
    #FILE = 'a.txt'
    OrgWordCount = 0
    CompressedWC = 0
    
    oo = SimplifyDepParse()
    oo.printable = 0
    
    with open(FILE, encoding='utf8') as f:
                        
        SENT = []
        COMMENT = []
                        
        
        for line in f:
            
            line = line.strip()
            if line.startswith('#'):
                COMMENT.append(line)
                continue
            
            if line.strip() == '':
                if len(SENT) == 0: 
                    continue
                
                OrgWordCount += len(SENT)
                
                RESULT = oo.simplify_dep_parse(SENT)
                
                RESULT = oo.merge_linear_dep(RESULT)
                
                CompressedWC += len(RESULT)
                
                RESULT = oo.assign_impt_factor(RESULT, 1)
                
                oo.print_dep_parse(SENT, RESULT)
                
                SENT = []
                COMMENT = []                
                continue
                
            w_list = line.split('\t')
           
            SENT.append(w_list)  
            
    oo.OutFile.close()

    print(OrgWordCount, CompressedWC, (CompressedWC/float(OrgWordCount)))
               