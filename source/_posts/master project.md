---
title: master project
top: false
cover: false
toc: true
mathjax: true
date: 2022-04-12 14:00
password:
summary:
tags:
- project
categories:
- Neruroscience
---

# April 12th

## First meeting 

data: https://openneuro.org/datasets/ds003682

The file we cares:

![image-20220411140935965](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204111409061.png)

We can use MNE to open it

What these files do:

https://mne.tools/stable/generated/mne.read_epochs.html

More templates can be found in:

https://github.com/tobywise/aversive_state_reactivation/blob/master/notebooks/templates/sequenceness_classifier_template.ipynb

# April 13th

## Session meeting 

General understanding:

Title: the utility of multi-task machine learning for decoding brain states

How to write a lab notebookL

![image-20220413104227189](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131042297.png)

![image-20220413104255487](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131042577.png)

Table of Contents

page numbers; date; title/subject/experiment

![image-20220413104725413](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131047482.png)

![image-20220413105233836](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131052901.png)

Gantt charts is good to help organise time:

![image-20220413110103327](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204131101425.png)

# April 18th

## Install MNE-Python via pip:

Update Anaconda via ` conda upgrade --all` and  `conda install anaconda=2021.10`


```
## Package Plan ##

  environment location: F:\Anaconda

  added / updated specs:
    - anaconda=2021.11


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    anaconda-2021.11           |           py38_0          18 KB
    anaconda-client-1.9.0      |   py38haa95532_0         170 KB
    anaconda-project-0.10.1    |     pyhd3eb1b0_0         218 KB
    appdirs-1.4.4              |     pyhd3eb1b0_0          12 KB
    arrow-0.13.1               |           py38_0          81 KB
    astroid-2.6.6              |   py38haa95532_0         314 KB
    astropy-4.3.1              |   py38hc7d831d_0         6.1 MB
    attrs-21.2.0               |     pyhd3eb1b0_0          46 KB
    autopep8-1.5.7             |     pyhd3eb1b0_0          43 KB
    babel-2.9.1                |     pyhd3eb1b0_0         5.5 MB
    beautifulsoup4-4.10.0      |     pyh06a4308_0          85 KB
    binaryornot-0.4.4          |     pyhd3eb1b0_1         351 KB
    bitarray-2.3.0             |   py38h2bbff1b_1         141 KB
    bleach-4.0.0               |     pyhd3eb1b0_0         113 KB
    bokeh-2.4.1                |   py38haa95532_0         7.6 MB
    ca-certificates-2021.10.26 |       haa95532_2         115 KB
    certifi-2021.10.8          |   py38haa95532_0         152 KB
    cffi-1.14.6                |   py38h2bbff1b_0         224 KB
    cfitsio-3.470              |       he774522_6         512 KB
    charset-normalizer-2.0.4   |     pyhd3eb1b0_0          35 KB
    click-8.0.3                |     pyhd3eb1b0_0          79 KB
    cloudpickle-2.0.0          |     pyhd3eb1b0_0          32 KB
    comtypes-1.1.10            |py38haa95532_1002         236 KB
    conda-pack-0.6.0           |     pyhd3eb1b0_0          29 KB
    contextlib2-0.6.0.post1    |     pyhd3eb1b0_0          13 KB
    cookiecutter-1.7.2         |     pyhd3eb1b0_0          86 KB
    cryptography-3.4.8         |   py38h71e12ea_0         637 KB
    curl-7.78.0                |       h86230a5_0         132 KB
    cython-0.29.24             |   py38h604cdb4_0         1.8 MB
    daal4py-2021.3.0           |   py38h757b272_0         7.6 MB
    dal-2021.3.0               |     haa95532_564        24.4 MB
    dask-2021.10.0             |     pyhd3eb1b0_0          19 KB
    dask-core-2021.10.0        |     pyhd3eb1b0_0         718 KB
    dataclasses-0.8            |     pyh6d0b6a4_7           8 KB
    debugpy-1.4.1              |   py38hd77b12b_0         2.6 MB
    decorator-5.1.0            |     pyhd3eb1b0_0          14 KB
    diff-match-patch-20200713  |     pyhd3eb1b0_0          35 KB
    distributed-2021.10.0      |   py38haa95532_0        1010 KB
    docutils-0.17.1            |   py38haa95532_1         696 KB
    et_xmlfile-1.1.0           |   py38haa95532_0          10 KB
    filelock-3.3.1             |     pyhd3eb1b0_1          12 KB
    flake8-3.9.2               |     pyhd3eb1b0_0         129 KB
    fonttools-4.25.0           |     pyhd3eb1b0_0         632 KB
    fsspec-2021.10.1           |     pyhd3eb1b0_0          96 KB
    gevent-21.8.0              |   py38h2bbff1b_1         1.4 MB
    greenlet-1.1.1             |   py38hd77b12b_0          82 KB
    heapdict-1.0.1             |     pyhd3eb1b0_0           8 KB
    html5lib-1.1               |     pyhd3eb1b0_0          91 KB
    idna-3.2                   |     pyhd3eb1b0_0          48 KB
    imagecodecs-2021.8.26      |   py38ha1f97ea_0         6.2 MB
    importlib-metadata-4.8.1   |   py38haa95532_0          40 KB
    importlib_metadata-4.8.1   |       hd3eb1b0_0          11 KB
    inflection-0.5.1           |   py38haa95532_0          12 KB
    intel-openmp-2021.4.0      |    haa95532_3556         2.2 MB
    intervaltree-3.1.0         |     pyhd3eb1b0_0          25 KB
    ipykernel-6.4.1            |   py38haa95532_1         192 KB
    ipython-7.29.0             |   py38hd4e2768_0        1017 KB
    ipywidgets-7.6.5           |     pyhd3eb1b0_1         105 KB
    isort-5.9.3                |     pyhd3eb1b0_0          83 KB
    itsdangerous-2.0.1         |     pyhd3eb1b0_0          18 KB
    jdcal-1.4.1                |     pyhd3eb1b0_0          10 KB
    jedi-0.18.0                |   py38haa95532_1         910 KB
    jinja2-time-0.2.0          |     pyhd3eb1b0_2          17 KB
    joblib-1.1.0               |     pyhd3eb1b0_0         211 KB
    jpeg-9d                    |       h2bbff1b_0         283 KB
    json5-0.9.6                |     pyhd3eb1b0_0          21 KB
    jsonschema-3.2.0           |     pyhd3eb1b0_2          47 KB
    jupyter_core-4.8.1         |   py38haa95532_0          90 KB
    jupyterlab-3.2.1           |     pyhd3eb1b0_1         3.6 MB
    jupyterlab_server-2.8.2    |     pyhd3eb1b0_0          46 KB
    keyring-23.1.0             |   py38haa95532_0          77 KB
    krb5-1.19.2                |       h5b6d351_0         697 KB
    lerc-3.0                   |       hd77b12b_0         120 KB
    libcurl-7.78.0             |       h86230a5_0         294 KB
    libdeflate-1.8             |       h2bbff1b_5          46 KB
    libwebp-1.2.0              |       h2bbff1b_0         643 KB
    libxml2-2.9.12             |       h0ad7f3c_0         1.5 MB
    llvmlite-0.37.0            |   py38h23ce68f_1        13.3 MB
    lz4-c-1.9.3                |       h2bbff1b_1         132 KB
    matplotlib-3.4.3           |   py38haa95532_0          29 KB
    matplotlib-base-3.4.3      |   py38h49ac443_0         5.5 MB
    matplotlib-inline-0.1.2    |     pyhd3eb1b0_2          12 KB
    menuinst-1.4.18            |   py38h59b6b97_0          96 KB
    mkl-2021.4.0               |     haa95532_640       114.9 MB
    mkl-service-2.4.0          |   py38h2bbff1b_0          51 KB
    mkl_fft-1.3.1              |   py38h277e83a_0         139 KB
    mkl_random-1.2.2           |   py38hf11a4ad_0         225 KB
    more-itertools-8.10.0      |     pyhd3eb1b0_0          47 KB
    munkres-1.1.4              |             py_0          13 KB
    nbconvert-6.1.0            |   py38haa95532_0         500 KB
    networkx-2.6.3             |     pyhd3eb1b0_0         1.3 MB
    nltk-3.6.5                 |     pyhd3eb1b0_0         979 KB
    notebook-6.4.5             |   py38haa95532_0         4.5 MB
    numba-0.54.1               |   py38hf11a4ad_0         3.3 MB
    numpy-1.20.3               |   py38ha4e8547_0          23 KB
    numpy-base-1.20.3          |   py38hc2deb75_0         4.2 MB
    olefile-0.46               |     pyhd3eb1b0_0          34 KB
    openjpeg-2.4.0             |       h4fc8c34_0         219 KB
    openpyxl-3.0.9             |     pyhd3eb1b0_0         164 KB
    openssl-1.1.1l             |       h2bbff1b_0         4.8 MB
    packaging-21.0             |     pyhd3eb1b0_0          36 KB
    pandas-1.3.4               |   py38h6214cd6_0         8.6 MB
    parso-0.8.2                |     pyhd3eb1b0_0          69 KB
    path-16.0.0                |   py38haa95532_0          38 KB
    path.py-12.5.0             |       hd3eb1b0_0           4 KB
    pathlib2-2.3.6             |   py38haa95532_2          36 KB
    patsy-0.5.2                |   py38haa95532_0         275 KB
    pillow-8.4.0               |   py38hd45dc43_0         911 KB
    pip-21.2.2                 |   py38haa95532_0         1.9 MB
    pkginfo-1.7.1              |   py38haa95532_0          60 KB
    poyo-0.5.0                 |     pyhd3eb1b0_0          17 KB
    prometheus_client-0.11.0   |     pyhd3eb1b0_0          47 KB
    prompt-toolkit-3.0.20      |     pyhd3eb1b0_0         259 KB
    prompt_toolkit-3.0.20      |       hd3eb1b0_0          12 KB
    pycodestyle-2.7.0          |     pyhd3eb1b0_0          41 KB
    pycurl-7.44.1              |   py38hcd4344a_1          67 KB
    pydocstyle-6.1.1           |     pyhd3eb1b0_0          36 KB
    pyerfa-2.0.0               |   py38h2bbff1b_0         360 KB
    pyflakes-2.3.1             |     pyhd3eb1b0_0          60 KB
    pygments-2.10.0            |     pyhd3eb1b0_0         725 KB
    pylint-2.9.6               |   py38haa95532_1         509 KB
    pyls-spyder-0.4.0          |     pyhd3eb1b0_0          11 KB
    pyodbc-4.0.31              |   py38hd77b12b_0          68 KB
    pyopenssl-21.0.0           |     pyhd3eb1b0_1          49 KB
    pyparsing-3.0.4            |     pyhd3eb1b0_0          81 KB
    pyrsistent-0.18.0          |   py38h196d8e1_0          89 KB
    pytest-6.2.4               |   py38haa95532_2         440 KB
    python-3.8.12              |       h6244533_0        16.0 MB
    python-dateutil-2.8.2      |     pyhd3eb1b0_0         233 KB
    python-lsp-black-1.0.0     |     pyhd3eb1b0_0           8 KB
    python-lsp-jsonrpc-1.0.0   |     pyhd3eb1b0_0          10 KB
    python-lsp-server-1.2.4    |     pyhd3eb1b0_0          41 KB
    python-slugify-5.0.2       |     pyhd3eb1b0_0          13 KB
    pytz-2021.3                |     pyhd3eb1b0_0         171 KB
    pyyaml-6.0                 |   py38h2bbff1b_1         146 KB
    pyzmq-22.2.1               |   py38hd77b12b_1         622 KB
    qdarkstyle-3.0.2           |     pyhd3eb1b0_0         337 KB
    qstylizer-0.1.10           |     pyhd3eb1b0_0          17 KB
    qtconsole-5.1.1            |     pyhd3eb1b0_0          98 KB
    qtpy-1.10.0                |     pyhd3eb1b0_0          35 KB
    regex-2021.8.3             |   py38h2bbff1b_0         302 KB
    requests-2.26.0            |     pyhd3eb1b0_0          59 KB
    rope-0.19.0                |     pyhd3eb1b0_0         126 KB
    scikit-image-0.18.3        |   py38hf11a4ad_0         9.1 MB
    scikit-learn-0.24.2        |   py38hf11a4ad_1         4.8 MB
    scikit-learn-intelex-2021.3.0|   py38haa95532_0          38 KB
    scipy-1.7.1                |   py38hbe87c03_2        13.8 MB
    seaborn-0.11.2             |     pyhd3eb1b0_0         218 KB
    send2trash-1.8.0           |     pyhd3eb1b0_1          19 KB
    setuptools-58.0.4          |   py38haa95532_0         779 KB
    singledispatch-3.7.0       |  pyhd3eb1b0_1001          12 KB
    six-1.16.0                 |     pyhd3eb1b0_0          18 KB
    sortedcontainers-2.4.0     |     pyhd3eb1b0_0          26 KB
    sphinx-4.2.0               |     pyhd3eb1b0_1         1.2 MB
    sphinxcontrib-htmlhelp-2.0.0|     pyhd3eb1b0_0          32 KB
    sphinxcontrib-serializinghtml-1.1.5|     pyhd3eb1b0_0          25 KB
    spyder-5.1.5               |   py38haa95532_1         9.4 MB
    spyder-kernels-2.1.3       |   py38haa95532_0         113 KB
    sqlalchemy-1.4.22          |   py38h2bbff1b_0         1.8 MB
    sqlite-3.36.0              |       h2bbff1b_0         780 KB
    sympy-1.9                  |   py38haa95532_0         9.3 MB
    tbb-2021.4.0               |       h59b6b97_0         148 KB
    tbb4py-2021.4.0            |   py38h59b6b97_0          71 KB
    tblib-1.7.0                |     pyhd3eb1b0_0          15 KB
    testpath-0.5.0             |     pyhd3eb1b0_0          81 KB
    text-unidecode-1.3         |     pyhd3eb1b0_0          65 KB
    threadpoolctl-2.2.0        |     pyh0d69192_0          16 KB
    tifffile-2021.7.2          |     pyhd3eb1b0_2         135 KB
    tinycss-0.4                |  pyhd3eb1b0_1002          39 KB
    tk-8.6.11                  |       h2bbff1b_0         3.3 MB
    tqdm-4.62.3                |     pyhd3eb1b0_1          83 KB
    traitlets-5.1.0            |     pyhd3eb1b0_0          89 KB
    typed-ast-1.4.3            |   py38h2bbff1b_1         135 KB
    typing_extensions-3.10.0.2 |     pyh06a4308_0          31 KB
    unidecode-1.2.0            |     pyhd3eb1b0_0         155 KB
    urllib3-1.26.7             |     pyhd3eb1b0_0         111 KB
    watchdog-2.1.3             |   py38haa95532_0         110 KB
    wcwidth-0.2.5              |     pyhd3eb1b0_0          26 KB
    werkzeug-2.0.2             |     pyhd3eb1b0_0         224 KB
    wheel-0.37.0               |     pyhd3eb1b0_1          33 KB
    whichcraft-0.6.1           |     pyhd3eb1b0_0          11 KB
    wincertstore-0.2           |   py38haa95532_2          15 KB
    xlsxwriter-3.0.1           |     pyhd3eb1b0_0         111 KB
    xlwings-0.24.9             |   py38haa95532_0         891 KB
    zipp-3.6.0                 |     pyhd3eb1b0_0          17 KB
    zope.interface-5.4.0       |   py38h2bbff1b_0         305 KB
    zstd-1.4.9                 |       h19a0ad4_0         478 KB
    ------------------------------------------------------------
                                           Total:       328.0 MB

The following NEW packages will be INSTALLED:

  arrow              pkgs/main/win-64::arrow-0.13.1-py38_0
  binaryornot        pkgs/main/noarch::binaryornot-0.4.4-pyhd3eb1b0_1
  cfitsio            pkgs/main/win-64::cfitsio-3.470-he774522_6
  charset-normalizer pkgs/main/noarch::charset-normalizer-2.0.4-pyhd3eb1b0_0
  conda-pack         pkgs/main/noarch::conda-pack-0.6.0-pyhd3eb1b0_0
  cookiecutter       pkgs/main/noarch::cookiecutter-1.7.2-pyhd3eb1b0_0
  daal4py            pkgs/main/win-64::daal4py-2021.3.0-py38h757b272_0
  dal                pkgs/main/win-64::dal-2021.3.0-haa95532_564
  dataclasses        pkgs/main/noarch::dataclasses-0.8-pyh6d0b6a4_7
  debugpy            pkgs/main/win-64::debugpy-1.4.1-py38hd77b12b_0
  fonttools          pkgs/main/noarch::fonttools-4.25.0-pyhd3eb1b0_0
  inflection         pkgs/main/win-64::inflection-0.5.1-py38haa95532_0
  jinja2-time        pkgs/main/noarch::jinja2-time-0.2.0-pyhd3eb1b0_2
  libwebp            pkgs/main/win-64::libwebp-1.2.0-h2bbff1b_0
  matplotlib-inline  pkgs/main/noarch::matplotlib-inline-0.1.2-pyhd3eb1b0_2
  munkres            pkgs/main/noarch::munkres-1.1.4-py_0
  poyo               pkgs/main/noarch::poyo-0.5.0-pyhd3eb1b0_0
  python-lsp-black   pkgs/main/noarch::python-lsp-black-1.0.0-pyhd3eb1b0_0
  python-lsp-jsonrpc pkgs/main/noarch::python-lsp-jsonrpc-1.0.0-pyhd3eb1b0_0
  python-lsp-server  pkgs/main/noarch::python-lsp-server-1.2.4-pyhd3eb1b0_0
  python-slugify     pkgs/main/noarch::python-slugify-5.0.2-pyhd3eb1b0_0
  qstylizer          pkgs/main/noarch::qstylizer-0.1.10-pyhd3eb1b0_0
  scikit-learn-inte~ pkgs/main/win-64::scikit-learn-intelex-2021.3.0-py38haa95532_0
  tbb4py             pkgs/main/win-64::tbb4py-2021.4.0-py38h59b6b97_0
  text-unidecode     pkgs/main/noarch::text-unidecode-1.3-pyhd3eb1b0_0
  tinycss            pkgs/main/noarch::tinycss-0.4-pyhd3eb1b0_1002
  unidecode          pkgs/main/noarch::unidecode-1.2.0-pyhd3eb1b0_0
  whichcraft         pkgs/main/noarch::whichcraft-0.6.1-pyhd3eb1b0_0

The following packages will be REMOVED:

  pyls-black-0.4.6-hd3eb1b0_0
  python-language-server-0.36.2-pyhd3eb1b0_0

The following packages will be UPDATED:

  anaconda                                   2021.05-py38_0 --> 2021.11-py38_0
  anaconda-client                              1.7.2-py38_0 --> 1.9.0-py38haa95532_0
  anaconda-project                       0.9.1-pyhd3eb1b0_1 --> 0.10.1-pyhd3eb1b0_0
  astroid                                2.5-py38haa95532_1 --> 2.6.6-py38haa95532_0
  astropy                              4.2.1-py38h2bbff1b_1 --> 4.3.1-py38hc7d831d_0
  attrs                                 20.3.0-pyhd3eb1b0_0 --> 21.2.0-pyhd3eb1b0_0
  autopep8                               1.5.6-pyhd3eb1b0_0 --> 1.5.7-pyhd3eb1b0_0
  babel                                  2.9.0-pyhd3eb1b0_0 --> 2.9.1-pyhd3eb1b0_0
  beautifulsoup4                         4.9.3-pyha847dfd_0 --> 4.10.0-pyh06a4308_0
  bitarray                             1.9.2-py38h2bbff1b_1 --> 2.3.0-py38h2bbff1b_1
  bleach                                 3.3.0-pyhd3eb1b0_0 --> 4.0.0-pyhd3eb1b0_0
  bokeh                                2.3.2-py38haa95532_0 --> 2.4.1-py38haa95532_0
  ca-certificates                      2021.4.13-haa95532_1 --> 2021.10.26-haa95532_2
  certifi                          2020.12.5-py38haa95532_0 --> 2021.10.8-py38haa95532_0
  cffi                                1.14.5-py38hcd4344a_0 --> 1.14.6-py38h2bbff1b_0
  click                                  7.1.2-pyhd3eb1b0_0 --> 8.0.3-pyhd3eb1b0_0
  cloudpickle                                    1.6.0-py_0 --> 2.0.0-pyhd3eb1b0_0
  comtypes                          1.1.9-py38haa95532_1002 --> 1.1.10-py38haa95532_1002
  cryptography                         3.4.7-py38h71e12ea_0 --> 3.4.8-py38h71e12ea_0
  curl                                    7.71.1-h2a8f88b_1 --> 7.78.0-h86230a5_0
  cython                             0.29.23-py38hd77b12b_0 --> 0.29.24-py38h604cdb4_0
  dask                                2021.4.0-pyhd3eb1b0_0 --> 2021.10.0-pyhd3eb1b0_0
  dask-core                           2021.4.0-pyhd3eb1b0_0 --> 2021.10.0-pyhd3eb1b0_0
  decorator                              5.0.6-pyhd3eb1b0_0 --> 5.1.0-pyhd3eb1b0_0
  distributed                       2021.4.0-py38haa95532_0 --> 2021.10.0-py38haa95532_0
  docutils                              0.17-py38haa95532_1 --> 0.17.1-py38haa95532_1
  et_xmlfile         pkgs/main/noarch::et_xmlfile-1.0.1-py~ --> pkgs/main/win-64::et_xmlfile-1.1.0-py38haa95532_0
  filelock                              3.0.12-pyhd3eb1b0_1 --> 3.3.1-pyhd3eb1b0_1
  flake8                                 3.9.0-pyhd3eb1b0_0 --> 3.9.2-pyhd3eb1b0_0
  fsspec                                 0.9.0-pyhd3eb1b0_0 --> 2021.10.1-pyhd3eb1b0_0
  gevent                              21.1.2-py38h2bbff1b_1 --> 21.8.0-py38h2bbff1b_1
  greenlet                             1.0.0-py38hd77b12b_2 --> 1.1.1-py38hd77b12b_0
  idna                                    2.10-pyhd3eb1b0_0 --> 3.2-pyhd3eb1b0_0
  imagecodecs                      2021.3.31-py38h5da4933_0 --> 2021.8.26-py38ha1f97ea_0
  importlib-metadata                  3.10.0-py38haa95532_0 --> 4.8.1-py38haa95532_0
  importlib_metadata                      3.10.0-hd3eb1b0_0 --> 4.8.1-hd3eb1b0_0
  intel-openmp                        2021.2.0-haa95532_616 --> 2021.4.0-haa95532_3556
  ipykernel                            5.3.4-py38h5ca1d4c_0 --> 6.4.1-py38haa95532_1
  ipython                             7.22.0-py38hd4e2768_0 --> 7.29.0-py38hd4e2768_0
  ipywidgets                             7.6.3-pyhd3eb1b0_1 --> 7.6.5-pyhd3eb1b0_1
  isort                                  5.8.0-pyhd3eb1b0_0 --> 5.9.3-pyhd3eb1b0_0
  itsdangerous                           1.1.0-pyhd3eb1b0_0 --> 2.0.1-pyhd3eb1b0_0
  jedi                                0.17.2-py38haa95532_1 --> 0.18.0-py38haa95532_1
  joblib                                 1.0.1-pyhd3eb1b0_0 --> 1.1.0-pyhd3eb1b0_0
  jpeg                                        9b-hb83a4c4_2 --> 9d-h2bbff1b_0
  json5                                          0.9.5-py_0 --> 0.9.6-pyhd3eb1b0_0
  jupyter_core                         4.7.1-py38haa95532_0 --> 4.8.1-py38haa95532_0
  jupyterlab                            3.0.14-pyhd3eb1b0_1 --> 3.2.1-pyhd3eb1b0_1
  jupyterlab_server                      2.4.0-pyhd3eb1b0_0 --> 2.8.2-pyhd3eb1b0_0
  keyring                             22.3.0-py38haa95532_0 --> 23.1.0-py38haa95532_0
  krb5                                    1.18.2-hc04afaa_0 --> 1.19.2-h5b6d351_0
  lerc                                     2.2.1-hd77b12b_0 --> 3.0-hd77b12b_0
  libcurl                                 7.71.1-h2a8f88b_1 --> 7.78.0-h86230a5_0
  libdeflate                                 1.7-h2bbff1b_5 --> 1.8-h2bbff1b_5
  libxml2                                 2.9.10-hb89e7f3_3 --> 2.9.12-h0ad7f3c_0
  llvmlite                            0.36.0-py38h34b8924_4 --> 0.37.0-py38h23ce68f_1
  lz4-c                                    1.9.3-h2bbff1b_0 --> 1.9.3-h2bbff1b_1
  matplotlib                           3.3.4-py38haa95532_0 --> 3.4.3-py38haa95532_0
  matplotlib-base                      3.3.4-py38h49ac443_0 --> 3.4.3-py38h49ac443_0
  menuinst                            1.4.16-py38he774522_1 --> 1.4.18-py38h59b6b97_0
  mkl                                 2021.2.0-haa95532_296 --> 2021.4.0-haa95532_640
  mkl-service                          2.3.0-py38h2bbff1b_1 --> 2.4.0-py38h2bbff1b_0
  mkl_fft                              1.3.0-py38h277e83a_2 --> 1.3.1-py38h277e83a_0
  mkl_random                           1.2.1-py38hf11a4ad_2 --> 1.2.2-py38hf11a4ad_0
  more-itertools                         8.7.0-pyhd3eb1b0_0 --> 8.10.0-pyhd3eb1b0_0
  nbconvert                                    6.0.7-py38_0 --> 6.1.0-py38haa95532_0
  networkx                                         2.5-py_0 --> 2.6.3-pyhd3eb1b0_0
  nltk                                   3.6.1-pyhd3eb1b0_0 --> 3.6.5-pyhd3eb1b0_0
  notebook                             6.3.0-py38haa95532_0 --> 6.4.5-py38haa95532_0
  numba                               0.53.1-py38hf11a4ad_0 --> 0.54.1-py38hf11a4ad_0
  numpy                               1.20.1-py38h34a8a5c_0 --> 1.20.3-py38ha4e8547_0
  numpy-base                          1.20.1-py38haf7ebc8_0 --> 1.20.3-py38hc2deb75_0
  openjpeg                                 2.3.0-h5ec785f_1 --> 2.4.0-h4fc8c34_0
  openpyxl                               3.0.7-pyhd3eb1b0_0 --> 3.0.9-pyhd3eb1b0_0
  openssl                                 1.1.1k-h2bbff1b_0 --> 1.1.1l-h2bbff1b_0
  packaging                               20.9-pyhd3eb1b0_0 --> 21.0-pyhd3eb1b0_0
  pandas                               1.2.4-py38hd77b12b_0 --> 1.3.4-py38h6214cd6_0
  parso                                          0.7.0-py_0 --> 0.8.2-pyhd3eb1b0_0
  path                                15.1.2-py38haa95532_0 --> 16.0.0-py38haa95532_0
  pathlib2                             2.3.5-py38haa95532_2 --> 2.3.6-py38haa95532_2
  patsy                                        0.5.1-py38_0 --> 0.5.2-py38haa95532_0
  pillow                               8.2.0-py38h4fa10fc_0 --> 8.4.0-py38hd45dc43_0
  pip                                 21.0.1-py38haa95532_0 --> 21.2.2-py38haa95532_0
  pkginfo                              1.7.0-py38haa95532_0 --> 1.7.1-py38haa95532_0
  prometheus_client                     0.10.1-pyhd3eb1b0_0 --> 0.11.0-pyhd3eb1b0_0
  prompt-toolkit                        3.0.17-pyh06a4308_0 --> 3.0.20-pyhd3eb1b0_0
  prompt_toolkit                          3.0.17-hd3eb1b0_0 --> 3.0.20-hd3eb1b0_0
  pycodestyle                            2.6.0-pyhd3eb1b0_0 --> 2.7.0-pyhd3eb1b0_0
  pycurl                            7.43.0.6-py38h7a1dbc1_0 --> 7.44.1-py38hcd4344a_1
  pydocstyle                             6.0.0-pyhd3eb1b0_0 --> 6.1.1-pyhd3eb1b0_0
  pyerfa                               1.7.3-py38h2bbff1b_0 --> 2.0.0-py38h2bbff1b_0
  pyflakes                               2.2.0-pyhd3eb1b0_0 --> 2.3.1-pyhd3eb1b0_0
  pygments                               2.8.1-pyhd3eb1b0_0 --> 2.10.0-pyhd3eb1b0_0
  pylint                               2.7.4-py38haa95532_1 --> 2.9.6-py38haa95532_1
  pyls-spyder                            0.3.2-pyhd3eb1b0_0 --> 0.4.0-pyhd3eb1b0_0
  pyodbc                              4.0.30-py38ha925a31_0 --> 4.0.31-py38hd77b12b_0
  pyopenssl                             20.0.1-pyhd3eb1b0_1 --> 21.0.0-pyhd3eb1b0_1
  pyparsing                              2.4.7-pyhd3eb1b0_0 --> 3.0.4-pyhd3eb1b0_0
  pyrsistent                          0.17.3-py38he774522_0 --> 0.18.0-py38h196d8e1_0
  pytest                               6.2.3-py38haa95532_2 --> 6.2.4-py38haa95532_2
  python                                   3.8.8-hdbf39b2_5 --> 3.8.12-h6244533_0
  python-dateutil                        2.8.1-pyhd3eb1b0_0 --> 2.8.2-pyhd3eb1b0_0
  pytz                                  2021.1-pyhd3eb1b0_0 --> 2021.3-pyhd3eb1b0_0
  pyyaml                               5.4.1-py38h2bbff1b_1 --> 6.0-py38h2bbff1b_1
  pyzmq                               20.0.0-py38hd77b12b_1 --> 22.2.1-py38hd77b12b_1
  qdarkstyle                                     2.8.1-py_0 --> 3.0.2-pyhd3eb1b0_0
  qtconsole                              5.0.3-pyhd3eb1b0_0 --> 5.1.1-pyhd3eb1b0_0
  qtpy                                           1.9.0-py_0 --> 1.10.0-pyhd3eb1b0_0
  regex                             2021.4.4-py38h2bbff1b_0 --> 2021.8.3-py38h2bbff1b_0
  requests                              2.25.1-pyhd3eb1b0_0 --> 2.26.0-pyhd3eb1b0_0
  rope                                          0.18.0-py_0 --> 0.19.0-pyhd3eb1b0_0
  scikit-image                        0.18.1-py38hf11a4ad_0 --> 0.18.3-py38hf11a4ad_0
  scikit-learn                        0.24.1-py38hf11a4ad_0 --> 0.24.2-py38hf11a4ad_1
  scipy                                1.6.2-py38h66253e8_1 --> 1.7.1-py38hbe87c03_2
  seaborn                               0.11.1-pyhd3eb1b0_0 --> 0.11.2-pyhd3eb1b0_0
  send2trash                             1.5.0-pyhd3eb1b0_1 --> 1.8.0-pyhd3eb1b0_1
  setuptools                          52.0.0-py38haa95532_0 --> 58.0.4-py38haa95532_0
  singledispatch                      3.6.1-pyhd3eb1b0_1001 --> 3.7.0-pyhd3eb1b0_1001
  six                pkgs/main/win-64::six-1.15.0-py38haa9~ --> pkgs/main/noarch::six-1.16.0-pyhd3eb1b0_0
  sortedcontainers                       2.3.0-pyhd3eb1b0_0 --> 2.4.0-pyhd3eb1b0_0
  sphinx                                 4.0.1-pyhd3eb1b0_0 --> 4.2.0-pyhd3eb1b0_1
  sphinxcontrib-htm~                     1.0.3-pyhd3eb1b0_0 --> 2.0.0-pyhd3eb1b0_0
  sphinxcontrib-ser~                     1.1.4-pyhd3eb1b0_0 --> 1.1.5-pyhd3eb1b0_0
  spyder                               4.2.5-py38haa95532_0 --> 5.1.5-py38haa95532_1
  spyder-kernels                      1.10.2-py38haa95532_0 --> 2.1.3-py38haa95532_0
  sqlalchemy                           1.4.7-py38h2bbff1b_0 --> 1.4.22-py38h2bbff1b_0
  sqlite                                  3.35.4-h2bbff1b_0 --> 3.36.0-h2bbff1b_0
  sympy                                  1.8-py38haa95532_0 --> 1.9-py38haa95532_0
  tbb                                     2020.3-h74a9793_0 --> 2021.4.0-h59b6b97_0
  testpath                               0.4.4-pyhd3eb1b0_0 --> 0.5.0-pyhd3eb1b0_0
  threadpoolctl                          2.1.0-pyh5ca1d4c_0 --> 2.2.0-pyh0d69192_0
  tifffile                            2021.4.8-pyhd3eb1b0_2 --> 2021.7.2-pyhd3eb1b0_2
  tk                                      8.6.10-he774522_0 --> 8.6.11-h2bbff1b_0
  tqdm                                  4.59.0-pyhd3eb1b0_1 --> 4.62.3-pyhd3eb1b0_1
  traitlets                              5.0.5-pyhd3eb1b0_0 --> 5.1.0-pyhd3eb1b0_0
  typed-ast                            1.4.2-py38h2bbff1b_1 --> 1.4.3-py38h2bbff1b_1
  typing_extensions                    3.7.4.3-pyha847dfd_0 --> 3.10.0.2-pyh06a4308_0
  urllib3                               1.26.4-pyhd3eb1b0_0 --> 1.26.7-pyhd3eb1b0_0
  watchdog                             1.0.2-py38haa95532_1 --> 2.1.3-py38haa95532_0
  werkzeug                               1.0.1-pyhd3eb1b0_0 --> 2.0.2-pyhd3eb1b0_0
  wheel                                 0.36.2-pyhd3eb1b0_0 --> 0.37.0-pyhd3eb1b0_1
  wincertstore                                   0.2-py38_0 --> 0.2-py38haa95532_2
  xlsxwriter                             1.3.8-pyhd3eb1b0_0 --> 3.0.1-pyhd3eb1b0_0
  xlwings                             0.23.0-py38haa95532_0 --> 0.24.9-py38haa95532_0
  zipp                                   3.4.1-pyhd3eb1b0_0 --> 3.6.0-pyhd3eb1b0_0
  zope.interface                       5.3.0-py38h2bbff1b_0 --> 5.4.0-py38h2bbff1b_0
  zstd                                     1.4.5-h04227a9_0 --> 1.4.9-h19a0ad4_0

The following packages will be DOWNGRADED:

  appdirs                                        1.4.4-py_0 --> 1.4.4-pyhd3eb1b0_0
  contextlib2                              0.6.0.post1-py_0 --> 0.6.0.post1-pyhd3eb1b0_0
  diff-match-patch                            20200713-py_0 --> 20200713-pyhd3eb1b0_0
  heapdict                                       1.0.1-py_0 --> 1.0.1-pyhd3eb1b0_0
  html5lib                                         1.1-py_0 --> 1.1-pyhd3eb1b0_0
  intervaltree                                   3.1.0-py_0 --> 3.1.0-pyhd3eb1b0_0
  jdcal                                          1.4.1-py_0 --> 1.4.1-pyhd3eb1b0_0
  jsonschema                                     3.2.0-py_2 --> 3.2.0-pyhd3eb1b0_2
  olefile                                         0.46-py_0 --> 0.46-pyhd3eb1b0_0
  path.py                                          12.5.0-0 --> 12.5.0-hd3eb1b0_0
  tblib                                          1.7.0-py_0 --> 1.7.0-pyhd3eb1b0_0
  wcwidth                                        0.2.5-py_0 --> 0.2.5-pyhd3eb1b0_0


Proceed ([y]/n)?
```

The Python version I am using is:

```
Python 3.9.12	
```

get the data:

`git clone https://github.com/OpenNeuroDatasets/ds003682`

Install MNE:

`conda install --channel=conda-forge mne-base`

# April 19th

## Second meeting 

`x_raw`

`x_raw.shape`

`y_raw`

`time = localiser epchoes`



**scikit-learn** is the package I should use



use PCA to reduce dimensions 

![image-20220419104136746](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204191041843.png)

**need to be learned: normalization, regularization**

lasso L1 = sets unpredictive features to 0

ridge L2 = minimises the weights on unpredictive features

elastic net L1/L2



`random_search randomizedsearchCV` to test the performance



neural network can be the best way for logistic leaning 



validation	

# April 24th

从连续的脑电图信号中提取一些特定时间窗口的信号，这些时间窗口可以称作为epochs.

由于EEG是连续收集的，要分析脑电事件相关的电位时，需要将信号"切分"成时间片段，这些时间片段被锁定到某个事件（例如刺激）中的时间片段。

文章中的MEG数据无效，可能是因为数据损坏

incomplete copying led to corrupted files



The following events are present in the data: 1, 2, 3, 4, 5, 32

event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
'Visual/Left': 3, 'Visual/Right': 4,
'smiley': 5, 'button': 32}



sklearn.cross_validation在1.9版本以后就被弃用了，1.9版本的以后的小伙伴可以用sklearn.model_selection就行了，后面一样的

# April 25th

## Install scikit-learn (sklearn)

use the command line 

`conda install -c anaconda scikit-learn`

在Anaconda Prompt中启动base环境: `activate base`

并在环境下安装jupyter notebook、numpy等模块

```
conda insatll tensorflow

conda install jupyter notebook

conda install scikit-learn

conda install scipy
```

how to choose the right algorithm 

https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

![flow chart of scikit](https://scikit-learn.org/stable/_static/ml_map.png)

## Learning of sklearn

1. 导入模块

```
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```

2. 创建数据

加载 `iris` 的数据，把属性存在 `X`，类别标签存在 `y`：

```
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
```

观察一下数据集，`X` 有四个属性，`y` 有 0，1，2 三类：

```
print(iris_X[:2, :])
print(iris_y)

"""
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
 """
```

把数据集分为训练集和测试集，其中 `test_size=0.3`，即测试集占总数据的 30%：

```
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)
```

可以看到分开后的数据集，顺序也被打乱，这样更有利于学习模型：

```
print(y_train)

"""
[2 1 0 1 0 0 1 1 1 1 0 0 1 2 1 1 1 0 2 2 1 1 1 1 0 2 2 0 2 2 2 2 2 0 1 2 2
 2 2 2 2 0 1 2 2 1 1 1 0 0 1 2 0 1 0 1 0 1 2 2 0 1 2 2 2 1 1 1 1 2 2 2 1 0
 1 1 0 0 0 2 0 1 0 0 1 2 0 2 2 0 0 2 2 2 1 2 0 0 2 1 2 0 0 1 2]
 """
```

3. 建立模型－训练－预测

定义模块方式 `KNeighborsClassifier()`， 用 `fit` 来训练 `training data`，这一步就完成了训练的所有步骤， 后面的 `knn` 就已经是训练好的模型，可以直接用来 `predict` 测试集的数据， 对比用模型预测的值与真实的值，可以看到大概模拟出了数据，但是有误差，是不会完完全全预测正确的。

```
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)

"""
[2 0 0 1 2 2 0 0 0 1 2 2 1 1 2 1 2 1 0 0 0 2 1 2 0 0 0 0 1 0 2 0 0 2 1 0 1
 0 0 1 0 1 2 0 1]
[2 0 0 1 2 1 0 0 0 1 2 2 1 1 2 1 2 1 0 0 0 2 1 2 0 0 0 0 1 0 2 0 0 2 1 0 1
 0 0 1 0 1 2 0 1]
 """
```

![image-20220425150724894](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204251507965.png)





## Succeed at drawing plot

```
import mne
import os
from mne.datasets import sample
import matplotlib.pyplot as plt

# sample的存放地址
data_path = sample.data_path()
# 该fif文件存放地址
fname = 'E:\Proj\Previous data\sample\MEG\sample\sub-001_localiser_sub-001_ses-01_task-AversiveLearningReplay_run-localiser_proc_ICA-epo.fif.gz'

epochs = mne.read_epochs(fname)

print(epochs.event_id)

picks = mne.pick_types(epochs.info, meg=True, ref_meg=False, exclude='bads')

epochs.plot(block=True)

epochs.plot_drop_log()

plt.show()
```

```
epochs = mne.read_epochs(fname)

evoked = epochs.average()
evoked.plot_topomap()

plt.show()
```
```
availabe_event = [1, 2, 3, 4, 5, 32]

for i in availabe_event:
    evoked_i = epochs[i].average(picks=picks)
    epochs_i = epochs[i]
    evoked_i.plot(time_unit='s')
    plt.show()

```

# April 26th

## MRI safety training for 2.5 hrs

## update Anaconda

`conda update conda`

`conda update anaconda`

`conda update --all`

done

Python version: 3.8.13-h6244533_0

## change the directory path of Jupyter notebook

1. `jupyter notebook --generate-config` get the config file. change the line `c.NotebookApp.notebook_dir = ''` to the directory I want

2. find jupyter notebook file, change the attributes. ![change the attributes](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204261614733.png)

## Link the local directory and Github

for the convenience of collaboration 

SSH connect public key (id_rsa.pub) was created before.

after create the directory, run the command in git:

```
git init
git add .
git git commit -m  "Comment"
git remote add origin "the url of directory"
git push -u origin main
```

## Journal club preparation

# April 27th

## Pycharm

Get Pycharm educational version via King’s email

install python 3.8 environment for running the code from https://github.com/tobywise/aversive_state_reactivation

run below code in pycharm

```
!conda create -n py38 python=3.8
!pip install mne
!pip install scikit-learn
!pip install plotly
!pip install cufflinks
!pip install networkx
!conda install numba
!pip install pyyaml
!pip install papermill
```

## Fixation

The function `joblib` does not exist in `sklearn.external` anymore.

Error occurs when run the function plot_confusion_matrix:

Deprecated since version 1.0: `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the following class methods: `from_predictions` or `from_estimator`.

use

```
ConfusionMatrixDisplay.from_predictions(y, y_pred)
```

instead of 

```
plot_confusion_matrix(mean_conf_mat[:n_stim, :n_stim], title='Normalised confusion matrix, accuracy = {0}'.format(np.round(mean_accuracy, 2)))
```

# April 28th

## Logistic regression cost function

The function is using the principle of maximum likelihood estimation to find the parameters $\theta$ for different models. At the meantime, a nice property is it is convex. So, this cost function is generally everyone use for fitting parameters in logistic regression.

![image-20220429010526272](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204290105356.png)

The way we are going to minimize the cost function is using gradient descent:

![image-20220429011210521](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202204290112574.png)

Other alternative optimization algorithms (no need to manually pick $\alpha$ studying rate:

1. Conjugate gradient
2. BFGS
3. L-BFGS

## Ways to storage trained model

set and train a simple SVC model

```
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X,y)
```
Storage:

1. pickle

```
import pickle #pickle模块

#保存Model(注:save文件夹要预先建立，否则会报错)
with open('save/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)

#读取Model
with open('save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    #测试读取后的Model
    print(clf2.predict(X[0:1]))
```

2. joblib(supposed to be faster when dealing with a large data, because the use of multiprocessing)

```
from sklearn.externals import joblib #jbolib模块

#保存Model(注:save文件夹要预先建立，否则会报错)
joblib.dump(clf, 'save/clf.pkl')

#读取Model
clf3 = joblib.load('save/clf.pkl')

#测试读取后的Model
print(clf3.predict(X[0:1]))

```

# April 29th

## Google Colab

Get the subscription of Google Colab and Google drive

clone data to google drive

Token: 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

connect Google Colab with VS code

## Regularization

We can use **regularization** to rescue the overfitting

There are two types of regularization:

- **L1 Regularization** (or Lasso Regularization)

  $Min$($$\sum_{i=1}^{n}{|y_i-w_ix_i|+p\sum_{i=1}^{n}|w_i|}$$)

- **L2 Regularization** (or Ridge Regularization)

  $Min$($$\sum_{i=1}^{n}{(y_i-w_ix_i)^2+p\sum_{i=1}^{n}w_i^2}$$)

where `p` is the tuning parameter which decides in what extent we want to penalize the model.

However, there is another method for combination

- **Elastic Net:** When L1 and L2 regularization combine together, it becomes the elastic net method, it adds a hyperparameter.

how to select:

| **S.No** | **L1 Regularization**                                   | **L2 Regularization**                                        |
| -------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| **1**    | Panelizes the sum of absolute value of weights.         | penalizes the sum of square weights.                         |
| **2**    | It has a sparse solution.                               | It has a non-sparse solution.                                |
| **3**    | It gives multiple solutions.                            | It has only one solution.                                    |
| **4**    | Constructed in feature selection.                       | No feature selection.                                        |
| **5**    | Robust to outliers.                                     | Not robust to outliers.                                      |
| **6**    | It generates simple and interpretable models.           | It gives more accurate predictions when the output variable is the function of whole input variables. |
| **7**    | Unable to learn complex data patterns.                  | Able to learn complex data patterns.                         |
| **8**    | Computationally inefficient over non-sparse conditions. | Computationally efficient because of having analytical solutions. |

# May 2nd

## Classifier centre

get a question: how to determine the classifier centre? 

for this case, it is around 20 to be at the middle/ top

GPU accelerations

It could be a little bit tricky to accelerate the calculation in sklearn with GPU. Here is a possible solution: https://developer.nvidia.com/blog/scikit-learn-tutorial-beginners-guide-to-gpu-accelerating-ml-pipelines/.  

## Third meeting:

possible deep learning package:

JAX, HAIKU

my aim is to inform bad-performance data with the training model of good-performance data in aims to increase the performance. One hallmark is to increase the mean accuracy of each cases as high as possible.

## Test

Mean accuracy:

```
[0.4288888888888889, 0.33666666666666667, 0.2777777777777778, 0.5022222222222222, 0.5066666666666667, 0.4245810055865922, 0.5577777777777778, 0.43222222222222223, 0.65, 0.47888888888888886, 0.3377777777777778, 0.4800469483568075, 0.27111111111111114, 0.37193763919821826, 0.4288888888888889, 0.40555555555555556, 0.46444444444444444, 0.7077777777777777, 0.5811111111111111, 0.4711111111111111, 0.4255555555555556, 0.5022222222222222, 0.45394006659267483, 0.38555555555555554, 0.6222222222222222, 0.4622222222222222, 0.35444444444444445, 0.47444444444444445]
```

# May 3rd

Successfully do confusion matrix.

pictures of each case are stored in the Github depository: 

https://github.com/ReveRoyl/MT_ML_Decoding/tree/main/Aversive_state_reactivation/notebooks/templates/save_folder

It takes around 36 minutes to run 28 cases. 

# May 10th

