# 의존 구문 분석기

주어진 문장에 대해 의존 구문 분석 수행 도구(ver. 0.5)

- 본 코드는 여러 문장이 들어있는 파일을 입력받아 형태소 분석을 수행한 후 의존 구문 분석을 진행
- 형태소 분석기는 Transformer 모델 사용
- Transformer 소스는 아래 깃허브 소스를 사용함
  - https://github.com/yseokchoi/attention-is-all-you-need-pytorch
- 의존 구문 분석은 Stack-Pointer Networks 모델 사용
- Stack-Pointer Networks 소스는 아래 깃허브 소스를 사용함
  - https://github.com/yseokchoi/KoreanDependencyParserusingStackPointer
- demo site: 오픈 예정



# 버전 정보

190416 version. 0.5



# 요구사항

python == 3.6

pytorch == 0.4

numpy == 1.15.4



# 입력 파일

- 한 줄에 한 문장 원칙

- | 예제(test.txt)                                               |
  | ------------------------------------------------------------ |
  | 그런 데라면 기를 쓰고 벗어나려고 하지 않았냐?<br />문득 그녀는 그가 어떤 가정에 입양되었는지 궁금해졌다.<br /> |



# 출력 파일

- CoNLL 포맷으로 파일 출력

- | 예제(result.txt)                                             |
  | :----------------------------------------------------------- |
  | #SENT: 그런 데라면 기를 쓰고 벗어나려고 하지 않았냐?<br/>1	그런	그런	DET	MM	_	2	det	_	_<br />2	데라면	데 이 라면	ADJ	NNB+VCP+EC	_	5	advcl	_	_<br />3	기를	기 를	NOUN	NNG+JKO	_	4	obj	_	_<br />4	쓰고	쓰 고	VERB	VV+EC	_	5	advcl	_	_<br />5	벗어나려고	벗어나 려고	VERB	VV+EC	_	0	root	_	_<br />6	하지	하 지	AUX	VX+EC	_	5	aux	_	_<br />7	않았냐?	않 았 냐 ?	AUX	VX+EP+EF+SF	_	6	aux	_	_<br /><br />#SENT: 문득 그녀는 그가 어떤 가정에 입양되었는지 궁금해졌다.<br />1	문득	문득	ADV	MAG	_	7	advmod	_	_<br />2	그녀는	그녀 는	PRON	NP+JX	_	7	nsubj	_	_<br />3	그가	그 가	PRON	NP+JKS	_	6	nsubj	_	_<br />4	어떤	어떤	DET	MM	_	5	det	_	_<br />5	가정에	가정 에	NOUN	NNG+JKB	_	6	obl	_	_<br />6	입양되었는지	입양 되 었 는지	VERB	NNG+XSV+EP+EC	_	7	advcl	_	_<br />7	궁금해졌다.	궁금 하 아 지 었 다 .	ADJ	XR+XSA+EC+VX+EP+EF+SF	_	0	root	_	_ |



# 실행 방법

python bin/main.py **-root_dir** ./ **-file_name** test.txt **-batch_size** 30 **-save_file** result.txt **-use_gpu**

```
Paraemters
 -root_dir: 입력 파일 들어있는 폴더 경로
 -file_name: 입력 파일명
 -save_file: 출력 파일명
 -batch_size: 배치 사이즈
 -use_gpu(optional): gpu 사용여부(default: False)
```

