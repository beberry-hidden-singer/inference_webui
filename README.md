# Beberry-Hidden-Singer Integrated WebUI
by karljeon44

## Setup
```shell
$ git clone https://github.com/beberry-hidden-singer/integrated_webui.git integrated-webui
```
### Expected project folder structure
```
ROOT
├───integrated-webui
│   ├───app.py
│   ├───DATA
│   ├───tmp
│   ├───summary
│   └───...
├───preliminary_preprocessing
│   ├───preprocess.py
│   └───...
├───ervc-v2
│   ├───infer_web.py
│   ├───logs
│   └───...
├───esovits
│   └───...
├───DDSP-Shallow-Diffusion
│   ├───preprocess.py
│   ├───train_diff.py
│   ├───infer.py
│   ├───exp
│   └───...
└───pretrain
    ├───contentvec
    ├───fcpe
    ├───hubert
    ├───rmvpe
    └───...
```

* assumed that a single python environment covers all the projects
  * developed and tested with Python 3.8.15 and Gradio 3.30.0 on Debian and macOS
  * see [requirements.txt](https://github.com/beberry-hidden-singer/integrated_webui/blob/main/requirements.txt)

## Launch webUI
1. from `ROOT`:
    ```shell
    $ python integrated-webui/app.py
    ```
2. from `integrated-webui`:
     ```shell
    $ python app.py
    ```

## LICENSE
see [here](https://github.com/beberry-hidden-singer/integrated_webui/blob/main/LICENSE)

## DISCLAIMER
개인편의를 위해 개발된 프로그램이며 사용함에 있어 따르는 어떠한 문제에 대해서도 책임을 지지 않습니다.

Use it at your own risk.
