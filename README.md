# Path Representation Learning Library
PRL library is an open-source library for deep learning researchers, especially for PRL.

# PRL Methods

This library currently supports the following methods:

|Method | Venues | Modality |
|--------|--------|--------|
| **PIM**  | [IJCAI 2021](https://ijcai-21.org/index.html)  | Path  |
| **TPR**   |[ICDE 2022](https://icde2022.ieeecomputer.my/) | Path |
| **LightPath**  | [KDD 2023](https://kdd.org/kdd2023/index.html)  | Path |
| **MM-Path**   |[KDD 2025](https://kdd2025.kdd.org/) | Path-Image |
| **Path-LLM** |  [WWW 2025](https://www2025.thewebconf.org/)  | Path-Text |
| **AutoPRL**   |[ICDE-26 Planned](https://icde2026.github.io/) | Path  |
| **GVCPath**   |[AAAI-26 Planned](https://aaai.org/conference/aaai/aaai-26/) | Path  |



# Dataset 

- Chengdu dataset is online available at [Google Drive](https://drive.google.com/file/d/1xc1TKmEQ0VQ7daA6KVPri9J9OmsYLai_/view?usp=drive_link).Thanks for DiDi.

- Harbin dataset is online availaabel at [Google Drive](https://drive.google.com/file/d/1TqupyC0LVqUtGfoPuXmIjm2VUke1lx0b/view?usp=drive_link). Thanks for authors of DeepGTT

To get path trajecotry, you can refer to map matching methods [FMM](https://github.com/cyang-kth/fmm) or [barefoot](https://github.com/boathit/barefoot).

# Citation

if you find this repo useful, please cite our papers.

<pre> <code>
@inproceedings{DBLP:conf/ijcai/YangGHT021,
	author       = {Sean Bin Yang and
	Chenjuan Guo and
	Jilin Hu and
	Jian Tang and
	Bin Yang},
	title        = {Unsupervised Path Representation Learning with Curriculum Negative
	Sampling},
	booktitle    = { {IJCAI} },
	pages        = {3286--3292},
	year         = {2021}
} </code> </pre>

<pre> <code>
@inproceedings{DBLP:conf/icde/YangGHYTJ22,
	author       = {Sean Bin Yang and
	Chenjuan Guo and
	Jilin Hu and
	Bin Yang and
	Jian Tang and
	Christian S. Jensen},
	title        = {Weakly-supervised Temporal Path Representation Learning with Contrastive
	Curriculum Learning},
	booktitle    = {{ICDE} },
	pages        = {2873--2885},
	year         = {2022}
} </code> </pre>

<pre> <code>
@inproceedings{DBLP:conf/kdd/YangHGYJ23,
	author       = {Sean Bin Yang and
	Jilin Hu and
	Chenjuan Guo and
	Bin Yang and
	Christian S. Jensen},
	title        = {LightPath: Lightweight and Scalable Path Representation Learning},
	booktitle    = {{KDD} },
	pages        = {2999--3010},
	year         = {2023}
} </code> </pre>

<pre> <code>
@inproceedings{DBLP:conf/kdd/0001CGGHY025,
  author       = {Ronghui Xu and
                  Hanyin Cheng and
                  Chenjuan Guo and
                  Hongfan Gao and
                  Jilin Hu and
                  Sean Bin Yang and
                  Bin Yang},
  title        = {MM-Path: Multi-modal, Multi-granularity Path Representation Learning},
  booktitle    = {{KDD} },
  pages        = {1703--1714},
  year         = {2025},
} </code> </pre>

<pre> <code>
@inproceedings{DBLP:conf/www/Wei0G0YH25,
  author       = {Yongfu Wei and
                  Yan Lin and
                  Hongfan Gao and
                  Ronghui Xu and
                  Sean Bin Yang and
                  Jilin Hu},
  title        = {Path-LLM: {A} Multi-Modal Path Representation Learning by Aligning
                  and Fusing with Large Language Models},
  booktitle    = { {WWW} },
  pages        = {2289--2298},
  year         = {2025},
} </code> </pre>


