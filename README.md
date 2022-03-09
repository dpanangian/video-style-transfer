# Overview

This package started as a complete refactor of the code provided by [rkhamilton](https://github.com/rkhamilton/) & [NerdyRodent](https://github.com/nerdyrodent/), which started out as a Katherine Crowson VQGAN+CLIP-derived Google colab notebook.

In addition to refactoring NerdyRodent's code into a more pythonic package to improve usability, this project includes the following noteable elements:
* LPIPS loss
* Fewshot-patch-based training to apply changes on videos


# Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
@article{huang2020rife,
  title={RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2011.06294},
  year={2020}
}
@Article{Texler20-SIG,
    author    = "Ond\v{r}ej Texler and David Futschik and Michal Ku\v{c}era and Ond\v{r}ej Jamri\v{s}ka and \v{S}\'{a}rka Sochorov\'{a} and Menglei Chai and Sergey Tulyakov and Daniel S\'{y}kora",
    title     = "Interactive Video Stylization Using Few-Shot Patch-Based Training",
    journal   = "ACM Transactions on Graphics",
    volume    = "39",
    number    = "4",
    pages     = "73",
    year      = "2020",
}
```

Katherine Crowson - <https://github.com/crowsonkb>  
NerdyRodent - <https://github.com/nerdyrodent/>
rkhamilton - <https://github.com/rkhamilton/>

