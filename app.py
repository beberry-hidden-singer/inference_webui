#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 8/30/23 2:16 PM
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import argparse
import datetime
import json
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
import re
import shutil
import sys
import threading
import traceback
import yaml
from pytz import timezone
from subprocess import Popen
from time import sleep

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from fairseq import checkpoint_utils

argparser = argparse.ArgumentParser('Beberry-Hidden-Singer Integrated WebUI Argparser')
argparser.add_argument('-p', '--port', type=int, default=7865, help='listen port')
argparser.add_argument('--debug', action='store_true', help='whether to log at DEBUG level')
args = argparser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(
  level=logging.DEBUG if args.debug else logging.INFO,
  format="[%(asctime)s][%(name)s][%(levelname)s]%(message)s",
  datefmt='%Y-%m-%d %H:%M:%S'
)

### python executable used to execute cmds
PYTHON = sys.executable


### default device, assuming a single node for inference
DEVICE = 'cpu'
if torch.cuda.is_available():
  DEVICE = 'cuda'
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
  DEVICE = 'mps'
DEVICE = torch.device(DEVICE)
logger.info("Using Device: %s", DEVICE)


### DEFAULT PATHS
ROOT_DIR = HOME_DIR = os.getcwd()
if 'integrated-webui' in os.path.basename(HOME_DIR):
  ROOT_DIR = os.path.abspath(os.path.join(HOME_DIR, os.pardir))
else:
  HOME_DIR = os.path.join(HOME_DIR, 'integrated-webui')

logger.debug("ROOT Dir: `%s`", ROOT_DIR)
logger.debug("HOME Dir: `%s`", HOME_DIR)

# UTIL DIRS
PRETRAIN_HOME = os.path.join(ROOT_DIR, 'pretrain')
CONTENTVEC_FPATH = os.path.join(PRETRAIN_HOME, 'contentvec', 'checkpoint_best_legacy_500.pt')
FCPE_FPATH = os.path.join(PRETRAIN_HOME, 'fcpe', 'fcpe.pt')
HUBERT_FPATH = os.path.join(PRETRAIN_HOME, 'hubert', 'hubert_base.pt')
RMVPE_FPATH = os.path.join(PRETRAIN_HOME, 'rmvpe', 'model.pt')

# preprocessed input audio data placed under `DATA` dir
DATA_DIR = os.path.join(HOME_DIR, 'DATA')
os.makedirs(DATA_DIR, exist_ok=True)
# inferred output audio, which will be shown as a playback on webui, placed under `tmp` dir
TMP_DIR = os.path.join(HOME_DIR, 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# model summary jsons
SUMMARY_DIR = os.path.join(HOME_DIR, 'summary')
os.makedirs(SUMMARY_DIR, exist_ok=True)
SUMMARY_FPATH_TEMPLATE = 'model-summary-%s-%s-%s.json'


### Gradio App utils
def get_latest_ckpt(ckpts):
  # assuming all ckpts are located in the same dir
  if len(ckpts) == 0:
    return ''
  steps = [int(re.sub('[^0-9]', "", os.path.basename(ckpt))) for ckpt in ckpts]
  return os.path.join(os.path.dirname(ckpts[0]), f'model_{max(steps)}.pt')


def if_done(done, p):
  while True:
    if p.poll() is None:
      sleep(0.5)
    else:
      break
  done[0] = True


def run_cmd(cmd, cwd=None):
  logger.info("Running cmd: `%s` (Working Dir: `%s`)", cmd, cwd)
  p = Popen(cmd, shell=True, cwd=cwd)
  done = [False]
  threading.Thread(target=if_done, args=(done, p,),).start()
  while True:
    sleep(1)
    if done[0]:
      break
  logger.info("Done.")



def clean():
  return {"value": "", "__type__": "update"}


def make_visible(num_outputs=1):
  return tuple([{'visible': True, "__type__": "update"} for _ in range(int(num_outputs))])


def make_invisible(num_outputs=1):
  return tuple([{'visible': False, "__type__": "update"} for _ in range(int(num_outputs))])


def refresh_choices(dir_fpath, target_ext='.wav', include_full_path=False):
  choices = []
  if os.path.exists(dir_fpath):
    for name in os.listdir(dir_fpath):
      if target_ext is None or target_ext == '':
        if include_full_path:
          choices.append(os.path.join(dir_fpath, name))
        else:
          choices.append(name)
      else:
        if name.endswith(target_ext):
          if include_full_path:
            choices.append(os.path.join(dir_fpath, name))
          else:
            choices.append(name)
  return {"choices": sorted(choices), "__type__": "update"}


def remove_dir(target_dirpath, retain_topmost_dir=False):
  shutil.rmtree(target_dirpath)
  if retain_topmost_dir:
    os.makedirs(target_dirpath, exist_ok=True)
  return f"Removed `{target_dirpath}`"


def remove_gradio_tmp_dir():
  # and make the confirm and no buttons invisible
  gradio_tmp_dir = '/tmp/gradio'
  try:
    msg = remove_dir(gradio_tmp_dir) if os.path.exists(gradio_tmp_dir) else "`/tmp/gradio` is already empty"
    return msg, *make_invisible(2)
  except:
    return f"Failed to remove Gradio TMP dir at `{gradio_tmp_dir}`.\nThis is not a critical error.", *make_invisible(2)


def remove_data_dir():
  # and make the confirm and no buttons invisible
  try:
    return remove_dir(DATA_DIR, retain_topmost_dir=True), *make_invisible(2)
  except:
    return f"Failed to remove `{DATA_DIR}`.\nThis is not a critical error.", *make_invisible(2)


def remove_summary_dir():
  # and make the confirm and no buttons invisible
  try:
    return remove_dir(SUMMARY_DIR, retain_topmost_dir=True), *make_invisible(2)
  except:
    return f"Failed to remove `{SUMMARY_DIR}`.\nThis is not a critical error.", *make_invisible(2)


def remove_tmp_dir():
  # and make the confirm and no buttons invisible
  try:
    return remove_dir(TMP_DIR, retain_topmost_dir=True), *make_invisible(2)
  except:
    return f"Failed to remove `{TMP_DIR}`.\nThis is not a critical error.", *make_invisible(2)


def update_audio(audio_fname):
  audio_fpath = audio_fname
  if not os.path.exists(audio_fpath):
    audio_fpath = os.path.join(DATA_DIR, audio_fname)
  return {'value': audio_fpath, "visible": True, "__type__": "update"}


def update_tmp_audio(audio_fname):
  audio_fpath = audio_fname
  if not os.path.exists(audio_fpath):
    audio_fpath = os.path.join(TMP_DIR, audio_fname)
  return {'value': audio_fpath, "visible": True, "__type__": "update"}


### Modules
class PreliminaryPreprocessing:

  HOME = os.path.join(ROOT_DIR, 'preliminary_preprocessing')
  EXISTS = False

  def __init__(self):

    if os.path.exists(self.HOME):
      self.EXISTS = True
      sys.path.append(self.HOME)

      # noinspection PyUnresolvedReferences
      from preprocess import preprocess

      # override main function
      self.preprocess = preprocess


  def __call__(self, audio_fpath, sample_rate, normalize, peak, loudness):
    if not self.EXISTS:
      return "`preliminary-preprocessing` has not been set up correctly", None

    if not audio_fpath:
      return "Valid audio fpath should be provided", None

    normalize = True if normalize == 'Yes' else False

    try:
      _, audio_fpaths = self.preprocess(
        audio_fpath, sample_rate, output_dirpath=DATA_DIR, do_normalize=normalize, peak=peak, loudness=loudness)

      prp_fpath = audio_fpaths[0]
      msg = f"Success.\n\nInput Audio to Upload: `{audio_fpath}`\n\nPreprocessed Audio to Upload: `{prp_fpath}`"

    except:
      msg = f'Failed to run preliminary preprocessing.\n{traceback.format_exc()}'
      prp_fpath = ""

    return msg, prp_fpath


class ERVC_V2:

  HOME = os.path.join(ROOT_DIR, 'ervc-v2')
  EXISTS = False

  def __init__(self):

    self.config = None
    self.hubert = None
    self.VC = None
    self.net_g = None
    self.tgt_sr = None

    self.weights_dir = os.path.join(self.HOME, 'weights')
    self.logs_dir = os.path.join(self.HOME, 'logs')

    self.load_audio = None
    self.merge = None

    self.net_g_init_fn = None
    self.vc_init_fn = None
    if os.path.exists(self.HOME):
      logger.info("Found Enhanced-RVC-V2 installation")
      self.EXISTS = True
      sys.path.extend([self.HOME, os.path.join(self.HOME, 'lib')])

      # noinspection PyUnresolvedReferences
      from lib.utils.config import Config
      self.config = Config()
      self.config.device = DEVICE

      # noinspection PyUnresolvedReferences
      from lib.utils.misc_utils import load_audio
      self.load_audio = load_audio

      # noinspection PyUnresolvedReferences
      from lib.utils.process_ckpt import merge
      self.merge = merge

      # noinspection PyUnresolvedReferences
      from lib.model.models import SynthesizerTrnMs768NSFsid
      self.net_g_init_fn = SynthesizerTrnMs768NSFsid

      # noinspection PyUnresolvedReferences
      from lib.model.vc_infer_pipeline import VC
      self.vc_init_fn = VC


  # utils
  def load_hubert(self):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([HUBERT_FPATH], suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(DEVICE)
    hubert_model = hubert_model.half()
    hubert_model.eval()
    self.hubert = hubert_model


  def refresh_weights(self):
    return refresh_choices(self.weights_dir, target_ext='.pth')


  def refresh_index_list(self):
    index_paths = []
    for root, dirs, files in os.walk(self.logs_dir, topdown=False):
      for name in files:
        if name.endswith(".index") and "trained" not in name:

          index_paths.append("%s/%s" % (os.path.basename(root), name))
    index_update = {"choices": sorted(index_paths), "__type__": "update"}
    return index_update


  def refresh(self):
    weights_update = self.refresh_weights()
    input_audios_update = refresh_choices(DATA_DIR)
    index_list_update = self.refresh_index_list()
    return weights_update, input_audios_update, index_list_update

  # core interaction
  def get_vc(self, sid, protect0, protect1):
    if not self.EXISTS:
      logger.info("`ervc-v2` has not been set up correctly")
      return

    try:
      person = "%s/%s" % (self.weights_dir, sid)
      logger.info("loading requested ervc-v2 model weights from %s" % person)

      cpt = torch.load(person, map_location="cpu")
      self.tgt_sr = cpt["config"][-2]
      logger.info("VC Target Sample Rate: %s", self.tgt_sr)
      n_spk = cpt["config"][-4] = cpt["weight"]["emb_g.weight"].shape[0]

      net_g = self.net_g_init_fn(*cpt["config"], is_half=self.config.is_half)
      del net_g.enc_q

      logger.info(net_g.load_state_dict(cpt["weight"], strict=False))
      net_g = net_g.eval().to(DEVICE)
      if self.config.is_half:
        net_g = net_g.half()
      else:
        net_g = net_g.float()

      self.VC = self.vc_init_fn(self.tgt_sr, self.config)
      self.net_g = net_g

      to_return_protect0 = {"visible": True, "value": protect0, "__type__": "update",}
      to_return_protect1 = {"visible": True, "value": protect1, "__type__": "update",}

      return {"visible": True, "maximum": n_spk, "__type__": "update"}, to_return_protect0, to_return_protect1

    except:
      logger.info('Failed to init ervc-v2 VC: %s' % traceback.format_exc())
      return {"visible": False, "__type__": "update"}, None, None


  def __call__(
          self,
          sid,
          input_audio_fpath,
          f0_up_key,
          f0_file,
          f0_method,
          file_index,
          index_rate,
          filter_radius,
          rms_mix_rate,
          protect,
  ):
    if not self.EXISTS:
      return "`ervc-v2` has not been set up correctly", (None, None)

    if not os.path.exists(input_audio_fpath):
      input_audio_fpath = os.path.join(DATA_DIR, input_audio_fpath)

    if input_audio_fpath is None or not os.path.exists(input_audio_fpath):
      return "Valid audio fpath should be provided", (None, None)

    if file_index is not None and len(file_index) > 0 and not os.path.exists(file_index):
      file_index = os.path.join(self.logs_dir, file_index)

    try:
      audio = self.load_audio(input_audio_fpath, 16000)
      audio_max = np.abs(audio).max() / 0.95
      if audio_max > 1:
        audio /= audio_max
      times = [0, 0, 0]

      if self.hubert is None:
        self.load_hubert()

      f0_up_key = int(f0_up_key)
      audio_opt = self.VC.pipeline(
        self.hubert,
        self.net_g,
        sid,
        audio,
        input_audio_fpath,
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        1,
        filter_radius,
        self.tgt_sr,
        0,
        rms_mix_rate,
        'v2',
        protect,
        f0_file=f0_file,
      )

      # now save audio at `tmp_dir`
      input_audio_fname = os.path.splitext(os.path.basename(input_audio_fpath))[0]
      input_audio_fname_aug = f'{input_audio_fname}-ervc-{f0_method}-fr{index_rate}-protect{protect}'
      if f0_up_key != 0:
        if f0_method > 0:
          f0_up_key = f'+{f0_up_key}'
        input_audio_fname_aug = f'{input_audio_fname_aug}-key{f0_up_key}'
      output_audio_fpath = os.path.join(TMP_DIR, f'{input_audio_fname_aug}.wav')
      sf.write(output_audio_fpath, audio_opt, self.tgt_sr)

      index_info = "Using index:%s." % file_index if os.path.exists(file_index) else "Index not used."
      return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss\nExported at `%s`" % (index_info, times[0], times[1], times[2], output_audio_fpath), (self.tgt_sr, audio_opt)

    except:
      msg = f'Failed: \n{traceback.format_exc()}'
      logger.info(msg)
      return msg, (None, None)


class DDSP_DIFF:

  HOME = os.path.join(ROOT_DIR, 'DDSP-Shallow-Diffusion')
  EXISTS = False

  COMBO_PY = os.path.join(HOME, 'combo.py')
  INFER_PY = os.path.join(HOME, 'infer.py')

  def __init__(self):

    self.exp_dir = os.path.join(self.HOME, 'exp')
    self.combo_dir = os.path.join(self.exp_dir, 'combo')

    if os.path.exists(self.HOME):
      self.EXISTS = True
      logger.info("Found DDSP-Shallow-Diffusion installation")

      os.makedirs(self.exp_dir, exist_ok=True)
      os.makedirs(self.combo_dir, exist_ok=True)


  def combo_ckpt_event(self, combo_ckpt):
    # if not os.path.exists(combo_ckpt):
    #   combo_ckpt = os.path.join(self.combo_dir, combo_ckpt)
    combo_config = os.path.join(self.combo_dir, combo_ckpt.replace('.ptc', '.json'))
    with open(combo_config, "r") as config:
      combo_config = yaml.safe_load(config)
    return clean(), {'value': combo_config['kstep_max'], '__type__': 'update'}


  def ddsp_ckpt_event(self):
    return clean(), {'value': 100, '__type__': 'update'}


  def refresh_exp(self):
    return refresh_choices(self.exp_dir, target_ext="")


  def refresh_ddsp_exp(self):
    exps_update = self.refresh_exp()
    exps_update['choices'] = [x for x in exps_update['choices'] if 'ddsp' in x]
    return exps_update


  def refresh_ckpts(self, exp):
    exp_dir = exp
    if exp is None or len(exp) == 0:
      return {'choices': [], "__type__": "update"}

    if not os.path.exists(exp):
      exp_dir = os.path.join(self.exp_dir, exp)

    ckpt_update = refresh_choices(exp_dir, target_ext='.pt')
    ckpt_update['value'] = get_latest_ckpt(ckpt_update['choices'])
    return ckpt_update


  def refresh_combo_ckpts(self):
    # combos have `.ptc` instead of `.pt` extension
    return refresh_choices(self.combo_dir, target_ext='.ptc')


  def refresh_naive_ckpts(self):
    exps_update = self.refresh_exp()
    exps = [x for x in exps_update['choices'] if 'naive' in x]
    choices = []
    for exp in exps:
      ckpts = self.refresh_ckpts(exp)['choices']
      for ckpt in ckpts:
        choices.append(os.path.join(exp, ckpt))
    return {'choices': sorted(choices), '__type__': 'update'}


  def refresh_diff_ckpts(self):
    exps_update = self.refresh_exp()
    exps = [x for x in exps_update['choices'] if 'naive' not in x and 'ddsp' not in x]
    choices = []
    for exp in exps:
      ckpts = self.refresh_ckpts(exp)['choices']
      for ckpt in ckpts:
        choices.append(os.path.join(exp, ckpt))
    return {'choices': sorted(choices), '__type__': 'update'}


  def refresh_diff(self):
    return self.refresh_naive_ckpts(), self.refresh_diff_ckpts()


  def refresh_ddsp(self, exp):
    ddsp_exp_update = self.refresh_ddsp_exp()

    ddsp_ckpt_update = {'choices': [], "__type__": "update"}
    if exp is not None and len(exp) > 0:
      ddsp_ckpt_update = self.refresh_ckpts(exp)

    return ddsp_exp_update, ddsp_ckpt_update


  def refresh(self):
    ddsp_exps = self.refresh_ddsp_exp()['choices']
    ddsp_ckpts = []
    for exp in ddsp_exps:
      ckpts = self.refresh_ckpts(exp)['choices']
      for ckpt in ckpts:
        ddsp_ckpts.append(os.path.join(exp, ckpt))
    ddsp_update = {'choices': ddsp_ckpts, '__type__': 'update'}
    combo_update = self.refresh_combo_ckpts()
    input_audios_update = refresh_choices(DATA_DIR)
    return ddsp_update, combo_update, input_audios_update


  def remove_combo_dir(self):
    if not self.EXISTS:
      return 'DDSP-Shallow-Diffuion installation not found', *make_invisible(2)
    try:
      return remove_dir(self.combo_dir, retain_topmost_dir=True), *make_invisible(2)
    except:
      return f"Failed to remove `{self.combo_dir}`.\nThis is not a critical error.", *make_invisible(2)



  def make_combo(self, naive_model, diff_model, combo_name):
    # always to `self.combo_dir` which is './exp/combo'
    if not os.path.exists(naive_model):
      naive_model = os.path.join(self.exp_dir, naive_model)
    if not os.path.exists(diff_model):
      diff_model = os.path.join(self.exp_dir, diff_model)

    cmd = f'{PYTHON} {self.COMBO_PY} -nmodel {naive_model} -model {diff_model} -exp {self.combo_dir} -n {combo_name}'
    run_cmd(cmd, cwd=self.HOME)

    # also record which naive and diff checkpoints were fused as a json
    # first, retrieve k_step_max from `config.yaml` inside the exp of `diff_model`
    diff_config = os.path.join(os.path.dirname(diff_model), 'config.yaml')
    with open(diff_config, "r") as config:
      config = yaml.safe_load(config)
    k_step_max = config['model']['k_step_max']

    record_fname = f'{combo_name}.json'
    record_fpath = os.path.join(self.combo_dir, record_fname)
    logger.info("Exporting Combo Record at %s" % record_fpath)
    with open(record_fpath, 'w') as f:
      json.dump({'naive': naive_model, 'shallow-diffusion': diff_model, 'kstep_max': k_step_max}, f)

    msg = f'Combo exported at `{self.combo_dir}` as `{combo_name}.pt`.\nRecord can be found in the same dir as `{record_fname}`'
    return msg


  def __call__(
          self,
          ddsp_ckpt,
          combo_ckpt,
          input_audio_fpath,
          spk_id=1,
          # spk_mix_dict="None",
          key=0,
          formant_shift_key=0,
          pitch_extractor='rmvpe',
          speedup='auto',
          method='auto',
          kstep=None,
          index_ratio=0,
  ):
    model_prefix = 'ddsp'
    if ddsp_ckpt is not None and len(ddsp_ckpt) > 0:
      # always prefer DDSP first
      if not os.path.exists(ddsp_ckpt):
        ddsp_ckpt = os.path.join(self.exp_dir, ddsp_ckpt)
      ckpt_flag = f'-ddsp {ddsp_ckpt}'
      if combo_ckpt is not None:
        logger.info("Both DDSP and Combo ckpts were provided; preferring DDSP..")
    elif combo_ckpt is not None and len(combo_ckpt) > 0:
      if not os.path.exists(combo_ckpt):
        combo_ckpt = os.path.join(self.combo_dir, combo_ckpt)
      model_prefix = 'combo'
      ckpt_flag = f'-nmodel {combo_ckpt}'
    else:
      return 'either `ddsp_ckpt` or `combo_ckpt` must be provided'

    if speedup == 0:
      speedup = 'auto'
    try:
      speedup = int(speedup)
    except:
      pass

    if kstep == 0:
      kstep = 'auto'
    if kstep == 'auto' and model_prefix == 'combo':
      # load from config
      combo_config = os.path.join(os.path.dirname(combo_ckpt), combo_ckpt.replace('.ptc', '.json'))
      with open(combo_config, "r") as config:
        combo_config = yaml.safe_load(config)
      kstep = combo_config['kstep_max']
    try:
      kstep = int(kstep)
    except:
      pass
    key = int(key)


    if not os.path.exists(input_audio_fpath):
      input_audio_fpath = os.path.join(DATA_DIR, input_audio_fpath)

    input_audio_fname = os.path.splitext(os.path.basename(input_audio_fpath))[0]
    input_audio_fname_aug = f'{input_audio_fname}-{model_prefix}-{pitch_extractor}'
    if key != 0:
      if key > 0:
        key = f'+{key}'
      input_audio_fname_aug = f'{input_audio_fname_aug}-key{key}'
    if spk_id > 1:
      input_audio_fname_aug = f'{input_audio_fname_aug}-spk{spk_id}'
    if speedup != 'auto':
      input_audio_fname_aug = f'{input_audio_fname_aug}-speedup{speedup}'
    if method != 'auto':
      input_audio_fname_aug = f'{input_audio_fname_aug}-{method}'
    if kstep is not None:
      input_audio_fname_aug = f'{input_audio_fname_aug}-kstep{kstep}'
    if model_prefix == 'combo' and index_ratio > 0:
      input_audio_fname_aug = f'{input_audio_fname_aug}-fr{index_ratio}'
    output_audio_fpath = os.path.join(TMP_DIR, f'{input_audio_fname_aug}.wav')

    cmd = " ".join([
      PYTHON,
      self.INFER_PY,
      ckpt_flag,
      f'-i {input_audio_fpath}',
      f'-o {output_audio_fpath}',
      f'-id {spk_id}',
      # f'-mix {spk_mix_dict}',
      f'-k {key}',
      f'-f {formant_shift_key}',
      f'-pe {pitch_extractor}',
      f'-speedup {speedup}',
      f'-method {method}',
      f'-kstep {kstep}',
      f'-ir {index_ratio}'
    ])
    try:
      run_cmd(cmd, cwd=self.HOME)
      msg = f"Inferred Audio stored at `{output_audio_fpath}`"
    except:
      msg = f'Failed: \n{traceback.format_exc()}'

    # output_audio_update = {'value': output_audio_fpath, 'type': '__update__'}
    return msg, output_audio_fpath


class ESovits:

  HOME = os.path.join(ROOT_DIR, 'esovits')
  EXISTS = False

  def __init__(self):
    if os.path.exists(self.HOME):
      logger.info("Found Enhanced-sovits-SVC installation")
      self.EXISTS = True

  def __call__(self):
    pass


### Module inits
PRP = PreliminaryPreprocessing()
ERVC = ERVC_V2()
DDSP_DIFF = DDSP_DIFF()
ESOVITS = ESovits()


### Build App
def build_readme_tab():
  with gr.Row():
    with open(os.path.join(HOME_DIR, 'README.md')) as f:
      readme = f.read()
    gr.Markdown(readme)


def build_preprocessing_tab():
  gr.Markdown(value="## [Preliminary Preprocessing](https://github.com/beberry-hidden-singer/preliminary_preprocessing) (for INFERENCE ONLY, as of Aug 30 2023)")
  gr.Markdown(value=f"Target Destination: `{DATA_DIR}`")
  gr.Markdown(value=f"* Project home: `{PRP.HOME}`")

  with gr.Group():
    with gr.Row():
      audio = gr.Audio(type="filepath", label="Input Audio to Preprocess", interactive=True)
      with gr.Column():
        sample_rate = gr.Radio(
          label="Target Sample Rate",
          choices=["32k", "40k", "44.1k", "48k"],
          value="44.1k",
          interactive=True,
        )
        with gr.Row():
          normalize = gr.Radio(
            label="Loudness Normalization",
            choices=["Yes", "No"],
            value="Yes",
            interactive=True,
          )
          # can't be changed at this point
          loudness_algorithm = gr.Textbox(
            label='Loudness Algorithm (fixed)',
            value='ITU-R BS.1770-4',
            interactive=False
          )
        with gr.Row():
          peak = gr.Slider(
            minimum=-20.,
            maximum=-0.1,
            label="True Peak (dB)",
            info="loudest at -0.1 dB",
            value=-1.0,
            interactive=True,
          )
          loudness = gr.Slider(
            minimum=-42.,
            maximum=-10.,
            label="LUFS (dB)",
            info='make louder at -14.0 dB',
            value=-23.0,
            interactive=True,
          )

  with gr.Group():
    # with gr.Row():
    with gr.Column():
      prp_button = gr.Button("Preprocess", variant='primary')
      with gr.Row():
        prp_output = gr.Audio(label="Preprocessed Audio")
        prp_info = gr.Textbox(label="Output Information")

    prp_button.click(
      fn=PRP,
      inputs=[audio, sample_rate, normalize, peak, loudness],
      outputs=[prp_info, prp_output]
    )


def build_ervc_tab():
  gr.Markdown(value="## [Enhanced RVC-V2](https://github.com/beberry-hidden-singer/enhanced-RVC-v2)")
  gr.Markdown(value=f"* Project home: `{ERVC.HOME}`")
  gr.Markdown(value=f"* [WARNING]: first run may take more time than usual as the model needs to set up first")

  gr.Markdown("### ERVC-V2 Inference")
  with gr.Row():
    sid0 = gr.Dropdown(label="Inference Voice", choices=[], info='Click REFRESH to update', interactive=True)
    refresh_button = gr.Button("REFRESH", variant="primary")
    spk_item = gr.Slider(
      minimum=0,
      maximum=109,
      step=1,
      label="Singer/Speaker ID",
      value=72,
      visible=False,
      interactive=True,
    )

  with gr.Group():
    with gr.Row():
      with gr.Column():
        input_audio = gr.Dropdown(
          label="Input Audio Path", choices=[], info='Click REFRESH to update', interactive=True)
        input_audio_playback = gr.Audio(type="filepath", interactive=True, label="Input Audio Playback", visible=False)
        input_audio.change(fn=update_audio, inputs=[input_audio], outputs=[input_audio_playback])

        f0method0 = gr.Radio(
          label="Pitch Extraction Algorithm",
          choices=["pm", "harvest", "crepe", "mangio", "rmvpe", "fcpe"],
          value="rmvpe",
          interactive=True,
        )
        filter_radius0 = gr.Slider(
          minimum=0,
          maximum=7,
          label="If >=3: apply median filtering to the harvested pitch results; can reduce breathiness.",
          value=3,
          step=1,
          interactive=True,
        )

      with gr.Column():
        file_index2 = gr.Dropdown(
          label="Auto-detected Index List",
          info='Click REFRESH to update',
          choices=[],
          interactive=True,
        )
        vc_transform0 = gr.Slider(
          minimum=-12,
          maximum=12,
          label="Pitch Translation in int Semi-tones",
          value=0,
          step=1,
          interactive=True,
        )

        refresh_button.click(
          fn=ERVC.refresh,
          inputs=[],
          outputs=[sid0, input_audio, file_index2]
        )

        index_rate1 = gr.Slider(
          minimum=0,
          maximum=1,
          label="Feature Ratio (controls accent strength; high value may result in artifacts)",
          value=0.33,
          interactive=True,
        )
      with gr.Column():
        rms_mix_rate0 = gr.Slider(
          minimum=0,
          maximum=1,
          label="Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume",
          value=0.,
          interactive=True,
        )
        protect0 = gr.Slider(
          minimum=0,
          maximum=0.5,
          label="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy",
          value=0.33,
          step=0.01,
          interactive=True,
        )
      f0_file = gr.File(label="Optional f0 curve file in .csv")
      but0 = gr.Button("Convert", variant="primary")
      with gr.Row():
        vc_output1 = gr.Textbox(label="Output Information")
        vc_output2 = gr.Audio(label="Inferred Audio")
      but0.click(
        fn=ERVC,
        inputs=[
          spk_item,
          input_audio,
          vc_transform0,
          f0_file,
          f0method0,
          file_index2,
          index_rate1,
          filter_radius0,
          rms_mix_rate0,
          protect0,
        ],
        outputs=[vc_output1, vc_output2],
      )
    sid0.change(fn=ERVC.get_vc, inputs=[sid0, protect0, protect0], outputs=[spk_item, protect0, protect0],)

  gr.Markdown("### Timbre Fusion")
  with gr.Group():
    with gr.Row():
      ckpt_a = gr.Textbox(label="Path to Model A", value="", interactive=True)
      ckpt_b = gr.Textbox(label="Path to Model B", value="", interactive=True)
      alpha_a = gr.Slider(
        minimum=0,
        maximum=1,
        label="Alpha for Model A",
        value=0.5,
        interactive=True,
      )
  with gr.Group():
    with gr.Row():
      sr_ = gr.Radio(
        label="Target Sample Rate",
        choices=["32k", "40k", "48k"],
        value="48k",
        interactive=True,
      )
      if_f0_ = gr.Radio(
        label="Whether models have Pitch Guidance",
        choices=["Yes", "No"],
        value="Yes",
        interactive=True,
      )
      info__ = gr.Textbox(
        label="Model Information", value="", max_lines=8, interactive=True
      )
      name_to_save0 = gr.Textbox(
        label="Fused model Name (without extension)",
        value="",
        max_lines=1,
        interactive=True,
      )
      version_2 = gr.Radio(
        label="Version",
        choices=["v1", "v2"],
        value="v2",
        interactive=True,
      )
  with gr.Group():
    with gr.Row():
      but6 = gr.Button("Fuse", variant="primary")
      info4 = gr.Textbox(label="Output Information", value="", max_lines=8)
    but6.click(
      ERVC.merge,
      [
        ckpt_a,
        ckpt_b,
        alpha_a,
        sr_,
        if_f0_,
        info__,
        name_to_save0,
        version_2,
      ],
      info4,
    )


def build_ddsp_diff_tab():
  gr.Markdown(value="## [DDSP-4.0 & Shallow Diffusion](https://github.com/beberry-hidden-singer/DDSP-shallow-diffusion)")
  gr.Markdown(value=f"* Project home: `{ERVC.HOME}`")

  gr.Markdown(value="### Make Naive + Shallow Diffusion Combo")
  gr.Markdown(value="* Naive checkpoints must have `naive` in its exp name to be detected")
  with gr.Group():
    with gr.Row():
      naive_ckpt = gr.Dropdown(
        label="Choose one Naive checkpoint",
        info='Click REFRESH to update',
        choices=[],
        visible=True,
        interactive=True,
      )

      diff_ckpt = gr.Dropdown(
        label="Choose one Shallow-Diffusion checkpoint",
        info='Click REFRESH to update',
        choices=[],
        visible=True,
        interactive=True,
      )
      combo_refresh_button = gr.Button("REFRESH", variant='primary')
      combo_refresh_button.click(
        fn=DDSP_DIFF.refresh_diff,
        outputs=[naive_ckpt, diff_ckpt],
      )

  with gr.Group():
    gr.Markdown(value="Combos will be exported at `combo` dir")
    with gr.Row():
      combo_name = gr.Textbox(
        label="Combo Name",
        value="combo",
        interactive=True
      )
      combo_button = gr.Button("Make Combo", variant='primary')
  combo_info = gr.Textbox(label="Combo Output Information")
  combo_button.click(
    fn=DDSP_DIFF.make_combo,
    inputs=[naive_ckpt, diff_ckpt, combo_name],
    outputs=[combo_info]
  )

  gr.Markdown(value="### DDSP/Naive + Shallow Diffusion Inference")
  gr.Markdown(value="* DDSP checkpoint must have `ddsp` in its exp name to be detected")
  gr.Markdown(value="* Naive + Shallow-Diffusion combo exp dir is fixed at `combo`")

  gr.Markdown("##### Choose either (a) DDSP or (b) Naive + Shallow-Diffusion Combo checkpoint")
  with gr.Group():
    with gr.Row():
      ddsp_ckpt = gr.Dropdown(
        label="[A] Choose a DDSP checkpoint",
        info='Click REFRESH to update',
        choices=[],
        visible=True,
        interactive=True,
      )
      combo_ckpt = gr.Dropdown(
        label="[B] Choose a Naive+Shallow-Diffusion Combo checkpoint",
        info='Click REFRESH to update',
        choices=[],
        visible=True,
        interactive=True,
      )
      # choosing one will automatically clear the other one
      # ddsp_ckpt.change(fn=clean, outputs=[combo_ckpt])
      # combo_ckpt.change(fn=clean, outputs=[ddsp_ckpt])
      infer_refresh_button = gr.Button("REFRESH", variant='primary')

  with gr.Group():
    with gr.Row():
      with gr.Column():
        input_audio = gr.Dropdown(
          label="Input Audio Path", choices=[], info='Click REFRESH to update', interactive=True)
        input_audio_playback = gr.Audio(type="filepath", interactive=True, label="Input Audio Playback", visible=False)
        input_audio.change(fn=update_audio, inputs=[input_audio], outputs=[input_audio_playback])
        infer_pe = gr.Radio(
          label="Pitch Extraction Algorithm",
          choices=["pm", "harvest", "crepe", "mangio", "rmvpe", "fcpe"],
          value="rmvpe",
          interactive=True,
        )

      with gr.Column():
        infer_spk = gr.Slider(
          minimum=1,
          maximum=77,
          label="Singer/Speaker ID",
          value=1,
          step=1,
          interactive=True,
        )
        pitch_translation = gr.Slider(
          minimum=-12,
          maximum=12,
          label="Pitch Translation in int Semi-tones",
          value=0,
          step=1,
          interactive=True,
        )
        formant_translation = gr.Slider(
          minimum=-12,
          maximum=12,
          label="Formant Shift in int Semi-tones",
          info="untested yet (as of Sep 3, 2023)",
          value=0,
          step=1,
          interactive=False,
        )
        method = gr.Radio(
          label="ODE Solver",
          choices=["ddim", "pndm", "dpm-solver", "unipc"],
          value="dpm-solver",
          interactive=True,
        )

      with gr.Column():
        kstep = gr.Slider(
          minimum=0,
          maximum=1000,
          label="Shallow-Diffusion Kstep (auto if 0)",
          info="at most max kstep per model",
          value=0,
          step=50,
          interactive=True,
        )
        speedup = gr.Slider(
          minimum=0,
          maximum=30,
          label="Inference Speedup (auto if 0)",
          info="There may be a perceived loss of sound quality when ‘speedup’ exceeds 20",
          value=0,
          step=1,
          interactive=True,
        )
        index_ratio = gr.Slider(
          minimum=0,
          maximum=1,
          label="Feature Retrieval Ratio",
          info="For Combo only; untested yet (as of Sep 3, 2023)",
          value=0.,
          interactive=False,
        )
      convert = gr.Button("Convert", variant="primary")

  combo_ckpt.change(
    fn=DDSP_DIFF.combo_ckpt_event,
    inputs=[combo_ckpt],
    outputs=[ddsp_ckpt, kstep]
  )
  ddsp_ckpt.change(
    fn=DDSP_DIFF.ddsp_ckpt_event,
    outputs=[combo_ckpt, kstep]
  )
  infer_refresh_button.click(
    fn=DDSP_DIFF.refresh,
    outputs=[ddsp_ckpt, combo_ckpt, input_audio],
  )

  with gr.Group():
    with gr.Row():
      infer_output_info = gr.Textbox(label="Output Information")
      infer_output_audio = gr.Audio(label="Inferred Audio")

  convert.click(
    fn=DDSP_DIFF,
    inputs=[
      ddsp_ckpt,
      combo_ckpt,
      input_audio,
      infer_spk,
      pitch_translation,
      formant_translation,
      infer_pe,
      speedup,
      method,
      kstep,
      index_ratio
    ],
    outputs=[infer_output_info, infer_output_audio]
  )

def build_listen_tab():
  with gr.Row():
    gr.Markdown(value="## Listen to your Audio")

  def listen_refresh_fn():
    dc = refresh_choices(DATA_DIR, '.wav')
    tc = refresh_choices(TMP_DIR, '.wav')
    return dc, tc

  def update_audio_and_info(audio_fname, use_tmp_dir=0):
    if use_tmp_dir:
      audio_update = update_tmp_audio(audio_fname)
    else:
      audio_update = update_audio(audio_fname)
    audio = sf.SoundFile(audio_update['value'])
    return audio_update, audio.extra_info

  target_choices, result_choices = listen_refresh_fn()

  with gr.Group():
    with gr.Row():
      with gr.Column():
        gr.Markdown("### Input Preprocessed Audio")
        gr.Markdown(f"* `{DATA_DIR}`")
        input_dropdown = gr.Dropdown(
          label="Input Audio Path", choices=target_choices['choices'], info='Click REFRESH to update', interactive=True)
        input_audio = gr.Audio(type="filepath", interactive=True, label="Input Audio Playback", visible=False)
        input_info = gr.Textbox(label="Input Audio Information")
        input_dropdown.change(fn=update_audio_and_info, inputs=[input_dropdown], outputs=[input_audio, input_info])

      with gr.Column():
        gr.Markdown("### Inferred Audio")
        gr.Markdown(f"* `{TMP_DIR}`")
        result_dropdown = gr.Dropdown(
          label="Result Audio Path", choices=result_choices['choices'], info='Click REFRESH to update', interactive=True)
        result_audio = gr.Audio(type="filepath", interactive=True, label="Result Audio Playback", visible=False)
        result_output = gr.Textbox(label="Result Audio Information")
        result_dropdown.change(fn=update_audio_and_info, inputs=[result_dropdown, gr.Number(value=1, visible=False)], outputs=[result_audio, result_output])

  refresh_button = gr.Button("REFRESH", variant="primary")
  refresh_button.click(fn=listen_refresh_fn, outputs=[input_dropdown, result_dropdown])


def build_summary_tab():
  with gr.Row():
    with gr.Column():
      gr.Markdown(value="## Model Summary per Guide")
      gr.Markdown(value="* For future reference & replicability")

  gr.Markdown(value="### 1. Basic Info")
  with gr.Group():
    with gr.Row():
      with gr.Column():
        gr.Markdown(value="#### Engineer Info")
        engineer = gr.Textbox(label="Name", value="Karl", interactive=True)
        contact = gr.Textbox(label="Contact", value="karljeon44@gmail.com", interactive=True)
        timez = gr.Radio(
          label="Timezone",
          choices=["Asia/Seoul", "America/New_York", "UTC"],
          value="America/New_York",
          interactive=True,
        )

      with gr.Column():
        gr.Markdown(value="#### Guest Info")
        guest = gr.Textbox(label="Name", value="베베리", interactive=True)
        nth = gr.Slider(
        minimum=0,
        maximum=10,
        label="Nth Event",
        value=0,
        step=1,
        interactive=True,
      )

      with gr.Column():
        gr.Markdown(value="#### Guide Info")
        guide = gr.Textbox(label="Name", value="", interactive=True)
        source = gr.Textbox(label="URL / Source", value="", interactive=True)
        identity = gr.Radio(
          label="Guide Name Mapping",
          choices=["Guide-A", "Guide-B", "Guide-C"],
          value='Guide-A',
          interactive=True
      )

  basic_info_fields = [engineer, contact, timez, guest, nth, guide, source, identity]

  gr.Markdown(value="### 2. Guide Data Preprocessing")
  with gr.Group():
    with gr.Row():
      with gr.Column():
        gr.Markdown(value="#### Preprocessing")
        uvr_choices = [
            'uvr-inst-hq1',
            'uvr-inst-hq2',
            'uvr-inst-hq3',
            'uvr-mdx-main',
            'uvr-mdx-main-340',
            'uvr-mdx-main-390',
            'uvr-mdx-main-406',
            'uvr-mdx-main-427',
            'uvr-mdx-main-438',
            'uvr-mdx-inst-main',
            'uvr-mdx-inst-full-292',
            'uvr-kim-vocal-1',
            'uvr-kim-vocal-2',
            'uvr-kim-inst',
            'uvr-vocft',
          ]
        vocal_remover = gr.Dropdown(
          label="Vocal Remover (choose `uvr-ensemble` for Ensemble)",
          value="mvsep",
          choices=['mvsep'] + uvr_choices + ['uvr-ensemble'],
          interactive=True
        )
        uvr_modules = gr.CheckboxGroup(
          label="UVR Modules",
          info="Select all used as part of UVR Ensemble (assuming AVG/AVG)",
          choices=uvr_choices,
          interactive=True,
          visible=False
        )
        def show_uvr_options(remover):
          if remover == 'uvr-ensemble':
            return {"visible": True, "__type__": "update"}
          return {"visible": False, "__type__": "update"}
        vocal_remover.change(show_uvr_options, inputs=[vocal_remover], outputs=[uvr_modules])

        dereverb = gr.Dropdown(
          label="De-Reverb",
          value="uvr-deecho-aggresive",
          choices=['uvr-deecho-normal', 'uvr-deecho-aggresive', 'uvr-dereverb', 'rx10-dereverb'],
          interactive=True
        )
        deecho_aggr = gr.Slider(
          minimum=1,
          maximum=20,
          label="UVR De-reverb/De-echo Aggression",
          value=3,
          step=1,
          interactive=True,
        )
        def show_aggr_options(remover):
          if 'uvr' in remover:
            return {"visible": True, "__type__": "update"}
          return {"visible": False, "__type__": "update"}
        dereverb.change(show_aggr_options, inputs=[dereverb], outputs=[deecho_aggr])

      with gr.Column():
        gr.Markdown(value="#### Postprocessing")
        rx10 = gr.CheckboxGroup(
          label="RX10 Modules",
          info="Select all used during RX10 post-processing",
          choices=[
            'Breath Control',
            'De-plosive',
            'Mouth De-Click',
            "De-Click",
            "De-Esser",
            "De-Reverb",
            "Voice-Denoise",
            "Spectral-Denoise",
            "High-pass Filter"
          ],
          interactive=True
        )
        tuning = gr.Radio(
          label="Tuning",
          choices=["Yes", "No"],
          value='No',
          interactive=True
        )
        tuning_file = gr.Textbox(
          label="Tuning Filename (.mpd if Melodyne)",
          value="",
          interactive=True,
          visible=False,
        )
        def show_tuning_file(x):
          if x == 'Yes':
            return {"visible": True, "__type__": "update"}
          return {"visible": False, "__type__": "update"}
        tuning.change(fn=show_tuning_file, inputs=[tuning], outputs=[tuning_file])

      with gr.Column():
        gr.Markdown(value="#### Loudness Control")
        with gr.Row():
          normalize = gr.Radio(
            label="Loudness Normalization",
            choices=["Yes", "No"],
            value="Yes",
            interactive=True,
          )
          loudness_algorithm = gr.Textbox(
            label='Loudness Algorithm (fixed)',
            value='ITU-R BS.1770-4',
            interactive=False
          )
        peak = gr.Slider(
          minimum=-20.,
          maximum=0.,
          label="True Peak (dB)",
          value=-1.0,
          interactive=True,
        )
        loudness = gr.Slider(
          minimum=-42.,
          maximum=-10.,
          label="LUFS (dB)",
          value=-23.0,
          interactive=True,
        )

  prp_fields = [vocal_remover, uvr_modules, dereverb, deecho_aggr, rx10, tuning, tuning_file, normalize, loudness_algorithm, peak, loudness]

  gr.Markdown(value="### 3. SVC Models")
  with gr.Row():
    with gr.Column():
      used_svcs = gr.CheckboxGroup(
        label="Used SVC Models",
        info="Click all to enable per-module summary below",
        choices=["ervc-v2", "DDSP-Shallow-Diffusion Combo", "Naive-Shallow-Diffusion Combo"],
        interactive=True,
      )
    with gr.Column():
      infer_key = gr.Slider(
        minimum=-12,
        maximum=12,
        label="Pitch Translation in int Semi-tones",
        value=0,
        step=1,
        interactive=True,
      )
      formant_key = gr.Slider(
        minimum=-12,
        maximum=12,
        label="Formant Shift in int Semi-tones",
        info="untested yet (as of Sep 3, 2023)",
        value=0,
        step=1,
        interactive=False,
      )

  svc_fields_common = [used_svcs, infer_key]

  # by default, all svc modules will initially be non-interactive
  with gr.Group():
    with gr.Row():
      with gr.Row():
        with gr.Column():
          gr.Markdown(value="#### ERVC-V2")
          with gr.Column():
            ervc_pe_training = gr.Radio(
              label="Pitch Extraction for Training",
              choices=["pm", "harvest", "crepe", "mangio", "rmvpe", "fcpe"],
              value="harvest",
              interactive=False,
            )
            ervc_pe_inference = gr.CheckboxGroup(
              label="Pitch Extraction(s) for Inference",
              choices=["pm", "harvest", "crepe", "mangio", "rmvpe", "fcpe"],
              value="rmvpe",
              interactive=False,
            )

        with gr.Column():
          gr.Markdown(value="#### DDSP-4.0 + Shallow Diffusion")
          with gr.Column():
            ddsp_pe_training = gr.Radio(
              label="Pitch Extraction for Training",
              choices=["pm", "harvest", "crepe", "mangio", "rmvpe", "fcpe"],
              value="rmvpe",
              interactive=False,
            )
            ddsp_pe_inference = gr.CheckboxGroup(
              label="Pitch Extraction(s) for Inference",
              choices=["pm", "harvest", "crepe", "mangio", "rmvpe", "fcpe"],
              value="rmvpe",
              interactive=False,
            )

        with gr.Column():
          gr.Markdown(value="#### Naive + Shallow Diffusion")
          with gr.Column():
            diff_pe_training = gr.Radio(
              label="Pitch Extraction for Training",
              choices=["pm", "harvest", "crepe", "mangio", "rmvpe", "fcpe"],
              value="rmvpe",
              interactive=False,
            )
            diff_pe_inference = gr.CheckboxGroup(
              label="Pitch Extraction(s) for Inference",
              choices=["pm", "harvest", "crepe", "mangio", "rmvpe", "fcpe"],
              value="rmvpe",
              interactive=False,
            )

  with gr.Group():
    with gr.Row():
      with gr.Column():
        ervc_filter = gr.Slider(
          minimum=0,
          maximum=7,
          label="Median Filtering Ratio",
          value=3,
          step=1,
          interactive=False,
        )
        ervc_fr = gr.Slider(
          minimum=0,
          maximum=1,
          label="Feature Retrieval Ratio",
          value=0.33,
          interactive=False,
        )
        ervc_envelope = gr.Slider(
          minimum=0,
          maximum=1,
          label="Volume Envelope Ratio",
          value=0.,
          interactive=False,
        )
        ervc_protect = gr.Slider(
          minimum=0,
          maximum=0.5,
          label="Protect Ratio",
          value=0.33,
          step=0.01,
          interactive=False,
        )
        ervc_checkpoints = gr.Dropdown(
          label="Checkpoints used (Choose all)",
          info='Click REFRESH to update',
          choices=[],
          interactive=False,
          multiselect=True,
          visible=True
        )
        ervc_refresh_button = gr.Button("REFRESH", interactive=False, variant='primary')
        ervc_refresh_button.click(
          fn=ERVC.refresh_weights,
          outputs=[ervc_checkpoints]
        )

      with gr.Column():
        ddsp_method = gr.CheckboxGroup(
          label="ODE Solver(s)",
          choices=["ddim", "pndm", "dpm-solver", "unipc"],
          value="dpm-solver",
          interactive=False,
        )
        ddsp_speedup = gr.Slider(
          minimum=1,
          maximum=100,
          label="Inference Speedup",
          value=10,
          step=1,
          interactive=False,
        )
        ddsp_exps = gr.Dropdown(
          label='DDSP EXP',
          info='Click REFRESH to update',
          choices=[],
          visible=True,
          interactive=False,
        )
        ddsp_checkpoints = gr.Dropdown(
          label="Checkpoints used (Choose one)",
          info='Choose DDSP EXP to update',
          choices=[],
          visible=True,
          interactive=False,
        )
        ddsp_exps.change(fn=DDSP_DIFF.refresh_ckpts, inputs=[ddsp_exps], outputs=[ddsp_checkpoints])
        ddsp_refresh_button = gr.Button("REFRESH", interactive=False, variant='primary')
        ddsp_refresh_button.click(
          fn=DDSP_DIFF.refresh_ddsp,
          inputs=[ddsp_exps],
          outputs=[ddsp_exps, ddsp_checkpoints]
        )

      with gr.Column():
        diff_method = gr.CheckboxGroup(
          label="ODE Solver(s)",
          choices=["ddim", "pndm", "dpm-solver", "unipc"],
          value="dpm-solver",
          interactive=False,
        )
        diff_kstep = gr.CheckboxGroup(
          label="Shallow-Diffusion Kstep(s)",
          choices=['100', '200', '1000'],
          interactive=False,
        )
        diff_speedup = gr.Slider(
          minimum=1,
          maximum=100,
          label="Inference Speedup",
          value=10,
          step=1,
          interactive=False,
        )
        diff_fr = gr.Slider(
          minimum=0,
          maximum=1,
          label="Feature Retrieval Ratio",
          info="For Combo only; untested yet (as of Sep 3, 2023)",
          value=0.,
          interactive=False,
        )
        diff_checkpoints = gr.Dropdown(
          label="Checkpoints used (Choose all)",
          info='Click REFRESH to update',
          choices=[],
          visible=True,
          multiselect=True,
          interactive=False,
        )
        diff_refresh_button = gr.Button("REFRESH", interactive=False, variant='primary')
        diff_refresh_button.click(
          fn=DDSP_DIFF.refresh_combo_ckpts,
          outputs=[diff_checkpoints]
        )

  ervc_fields = [ervc_pe_training, ervc_pe_inference, ervc_filter, ervc_fr, ervc_envelope, ervc_protect, ervc_checkpoints, ervc_refresh_button] # 8
  ddsp_fields = [ddsp_pe_training, ddsp_pe_inference, ddsp_method, ddsp_speedup, ddsp_exps, ddsp_checkpoints, ddsp_refresh_button] # 7
  diff_fields = [diff_pe_training, diff_pe_inference, diff_method, diff_kstep, diff_speedup, diff_fr, diff_checkpoints, diff_refresh_button] # 8
  # diff_fields = [diff_pe_training, diff_pe_inference, diff_method, diff_kstep, diff_speedup, diff_checkpoints, diff_refresh_button] # 7
  svc_fields_per = ervc_fields + ddsp_fields + diff_fields

  def update_svc_interactive(used_list):
    default = {"interactive": False, "__type__": "update"}
    update = {"interactive": True, "__type__": "update"}

    num_ervc_fields = len(ervc_fields)
    num_ddsp_fields = len(ddsp_fields)
    num_diff_fields = len(diff_fields)
    total_num_fields = num_ervc_fields + num_ddsp_fields + num_diff_fields

    updates = [default] * total_num_fields
    for used_svc in used_list:
      used_svc = used_svc.lower()
      if 'ervc' in used_svc:
        updates[0:num_ervc_fields] = [update] * num_ervc_fields
      elif 'ddsp' in used_svc:
        updates[num_ervc_fields:num_ervc_fields+num_ddsp_fields] = [update] * num_ddsp_fields
      elif 'naive' in used_svc:
        updates[num_ervc_fields+num_ddsp_fields:] = [update] * num_diff_fields
    return updates

  used_svcs.change(
    fn=update_svc_interactive,
    inputs=[used_svcs],
    outputs=svc_fields_per
  )

  def build_summary_dict(
          # engineer-info
          engineer_,
          contact_,
          timezone_,
          # guest-info
          guest_,
          nth_,
          # guide-info
          guide_,
          source_,
          identity_,
          # pre-processing
          vocal_remover_,
          uvr_modules_,
          dereverb_,
          deecho_aggr_,
          # post-processing
          rx10_,
          tuning_,
          tuning_file_,
          # loudness-control
          normalize_,
          loudness_algorithm_,
          peak_,
          loudness_,
          # SVC
          used_svcs_,
          infer_key_,
          # ERVC-V2
          ervc_pe_training_,
          ervc_pe_inference_,
          ervc_filter_,
          ervc_fr_,
          ervc_envelope_,
          ervc_protect_,
          ervc_ckpts_,
          # DDSP
          ddsp_pe_training_,
          ddsp_pe_inference_,
          ddsp_method_,
          ddsp_speedup_,
          ddsp_exp_,
          ddsp_checkpoint_,
          # Shallow-DIFF
          diff_pe_training_,
          diff_pe_inference_,
          diff_method_,
          diff_kstep_,
          diff_speedup_,
          diff_fr_,
          diff_checkpoint_,
  ):
    if vocal_remover_ != 'uvr-ensemble':
      uvr_modules_ = 'n/a'
    if 'rx10' in dereverb_:
      deecho_aggr_ = 'n/a'
    if len(rx10_) == 0:
      rx10_ = 'n/a'
    tuning_ = tuning_file_ if tuning_ == 'Yes' else 'n/a'

    if normalize_ == 'No':
      loudness_dict = 'n/a'
    else:
      loudness_dict = {
        'Algorithm': loudness_algorithm_,
        'True Peak (dB)': peak_,
        'LUFS (dB)': loudness_,
      }

    if 'ervc-v2' in used_svcs_:
      ervc_dict = {
        'Pitch Extraction for Training': ervc_pe_training_,
        'Pitch Extraction(s) for Inference': ervc_pe_inference_,
        'Median Filtering Ratio': ervc_filter_,
        'Feature Retrieval Ratio': ervc_fr_,
        'Volume Envelope Ratio': ervc_envelope_,
        'Protect Ratio': ervc_protect_,
        'Checkpoints Used': ervc_ckpts_
      }
    else:
      ervc_dict = 'n/a'

    if 'DDSP-Shallow-Diffusion Combo' in used_svcs_:
      ddsp_dict = {
        'Pitch Extraction for Training': ddsp_pe_training_,
        'Pitch Extraction(s) for Inference': ddsp_pe_inference_,
        'ODE Solver': ddsp_method_,
        'Diffusion Speedup': ddsp_speedup_,
        'Checkpoints Used': os.path.join(ddsp_exp_, ddsp_checkpoint_)
      }
    else:
      ddsp_dict = 'n/a'

    if 'Naive-Shallow-Diffusion Combo' in used_svcs_:
      # get naive / shallow diff ckpts for each diff_checkpoint
      diff_ckpt_dict = dict()
      for diff_ckpt in diff_checkpoint_:
        diff_ckpt_record = os.path.join(DDSP_DIFF.combo_dir, f'{os.path.splitext(diff_ckpt)[0]}.json')
        with open(diff_ckpt_record) as f:
          diff_record = json.load(f)
          kstep = diff_record.pop('kstep_max', 'auto')
          common_prefix = os.path.commonprefix(list(diff_record.values()))
          for k,v in diff_record.items():
            diff_record[k] = os.path.relpath(v, common_prefix)
          diff_record['kstep'] = kstep
          diff_ckpt_dict[diff_ckpt] = diff_record

      diff_dict = {
        'Pitch Extraction for Training': diff_pe_training_,
        'Pitch Extraction(s) for Inference': diff_pe_inference_,
        'ODE Solver': diff_method_,
        'Diffusion Kstep': diff_kstep_,
        'Diffusion Speedup': diff_speedup_,
        'Feature Retrieval Ratio': diff_fr_,
        'Checkpoints Used': diff_ckpt_dict
      }
    else:
      diff_dict = 'n/a'

    cur_time = datetime.datetime.now(timezone(timezone_))
    summary = {
      "Automatically Generated on": cur_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
      'Basic-Information': {
        'Engineer-Info': {
          'Name': engineer_,
          'Contact': contact_,
        },
        'Guest-Info': {
          'Name': guest_,
          'Nth Event': nth_,
        },
        'Guide-Info': {
          'Name': guide_,
          'URL/Source': source_,
          'Mappping': identity_,
        }
      },
      'Data-Preprocessing': {
        'Preprocessing': {
          'MR-Remover': vocal_remover_,
          'UVR-Modules for Ensemble': uvr_modules_,
          'Dereverb': dereverb_,
          'UVR-VR Architecture Aggression': deecho_aggr_,
        },
        'Postprocessing': {
          'RX10 Post-processing': rx10_,
          'Tuning': tuning_,
        },
        'Loudness-Control': loudness_dict
      },
      'SVC-Models': {
        'Pitch Transpose (Semi-tones)': infer_key_,
        'ervc-v2': ervc_dict,
        'DDSP-4.0': ddsp_dict,
        'Naive-Shallow-Diffusion Combo': diff_dict
      }
    }

    today = cur_time.strftime('%Y_%m_%d')
    summary_fname = SUMMARY_FPATH_TEMPLATE % (guide_, guest_, today)
    summary_fpath = os.path.join(SUMMARY_DIR, summary_fname)
    if os.path.exists(summary_fpath):
      logger.info("Overwriting an already existing summary at `%s`", summary_fpath)
    msg = f"Summary exported at `{summary_fpath}`\nYou can also Copy-Paste and save this as `{summary_fname}`"
    with open(summary_fpath, 'w') as f:
      json.dump(summary, f, indent=4, ensure_ascii=False)
    return {"value": summary, "__type__": "update"}, {"value": msg, "__type__": "update"}

  generate_button = gr.Button("Generate Summary in JSON", variant="primary")
  summary_json = gr.Json(label="Summary")
  summary_info = gr.Textbox(label="Summary Information")
  generate_button.click(
    fn=build_summary_dict,
    inputs=basic_info_fields + prp_fields + svc_fields_common + ervc_fields[:-1] + ddsp_fields[:-1] + diff_fields[:-1],
    outputs=[summary_json, summary_info]
  )


def build_purge_tab():
  gr.Markdown(value="## Purge temporary files and dirs")
  gr.Markdown(value="* **[WARNING] take care before proceeding; purged files cannot be restored.**".upper())

  with gr.Group():
    with gr.Row():
      with gr.Column():
        gr.Markdown(f"### 1. Remove tmp dir at `./tmp`")
        gr.Markdown(f"* contains all inferred audio by SVC models")
        with gr.Column():
          purge_tmp_dir_button = gr.Button("Purge tmp dir", variant='primary')
          with gr.Row():
            purge_tmp_dir_confirm_button = gr.Button("Confirm?", visible=False, variant="primary")
            purge_tmp_dir_no_button = gr.Button("NO", visible=False, variant="secondary")
          purge_tmp_dir_button.click(
            fn=make_visible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_tmp_dir_confirm_button, purge_tmp_dir_no_button]
          )
          purge_tmp_dir_info = gr.Textbox(label="")
          purge_tmp_dir_confirm_button.click(
            fn=remove_tmp_dir,
            outputs=[purge_tmp_dir_info, purge_tmp_dir_confirm_button, purge_tmp_dir_no_button]
          )
          purge_tmp_dir_no_button.click(
            fn=make_invisible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_tmp_dir_confirm_button, purge_tmp_dir_no_button]
          )

      with gr.Column():
        gr.Markdown(f"### 2. Remove DATA dir at `./DATA`")
        gr.Markdown(f"* contains all preprocessed input audio")
        with gr.Column():
          purge_data_dir_button = gr.Button("Purge DATA dir", variant='primary')
          with gr.Row():
            purge_data_dir_confirm_button = gr.Button("Confirm?", visible=False, variant="primary")
            purge_data_dir_no_button = gr.Button("NO", visible=False, variant="secondary")
          purge_data_dir_button.click(
            fn=make_visible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_data_dir_confirm_button, purge_data_dir_no_button]
          )
          purge_data_dir_info = gr.Textbox(label="")
          purge_data_dir_confirm_button.click(
            fn=remove_data_dir,
            outputs=[purge_data_dir_info, purge_data_dir_confirm_button, purge_data_dir_no_button]
          )
          purge_data_dir_no_button.click(
            fn=make_invisible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_data_dir_confirm_button, purge_data_dir_no_button]
          )

      with gr.Column():
        gr.Markdown(f"### 3. Remove summary dir at `./summary`")
        gr.Markdown(f"* contains model summaries")
        with gr.Column():
          purge_summary_button = gr.Button("Purge Summary dir", variant='primary')
          with gr.Row():
            purge_summary_confirm_button = gr.Button("Confirm?", visible=False, variant="primary")
            purge_summary_no_button = gr.Button("NO", visible=False, variant="secondary")
          purge_summary_button.click(
            fn=make_visible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_summary_confirm_button, purge_summary_no_button]
          )
          purge_summary_info = gr.Textbox(label="")
          purge_summary_confirm_button.click(
            fn=remove_summary_dir,
            outputs=[purge_summary_info, purge_summary_confirm_button, purge_summary_no_button]
          )
          purge_summary_no_button.click(
            fn=make_invisible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_summary_confirm_button, purge_summary_no_button]
          )

  with gr.Group():
    with gr.Row():
      with gr.Column():
        gr.Markdown(f"### 4. Remove DDSP-Shallow-Diffusion combo dir at `DDSP-Shallow-Diffusion/exp/combo`")
        gr.Markdown(f"* contains Naive + Shallow Diffusion Combo models")
        with gr.Column():
          purge_combo_button = gr.Button("Purge Combo dir", variant='primary')
          with gr.Row():
            purge_combo_confirm_button = gr.Button("Confirm?", visible=False, variant="primary")
            purge_combo_no_button = gr.Button("NO", visible=False, variant="secondary")
          purge_combo_button.click(
            fn=make_visible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_combo_confirm_button, purge_combo_no_button]
          )
          purge_combo_info = gr.Textbox(label="")
          purge_combo_confirm_button.click(
            fn=DDSP_DIFF.remove_combo_dir,
            outputs=[purge_combo_info, purge_combo_confirm_button, purge_combo_no_button]
          )
          purge_combo_no_button.click(
            fn=make_invisible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_combo_confirm_button, purge_combo_no_button]
          )

      with gr.Column():
        gr.Markdown(f"### 5. Remove gradio tmp dir at `/tmp/gradio`")
        gr.Markdown(f"* contains all input audio uploaded for preprocessing")
        with gr.Column():
          purge_gradio_tmp_button = gr.Button("Purge Gradio tmp Cache dir", variant='primary')
          with gr.Row():
            purge_gradio_tmp_confirm_button = gr.Button("Confirm?", visible=False, variant="primary")
            purge_gradio_tmp_no_button = gr.Button("NO", visible=False, variant="secondary")
          purge_gradio_tmp_button.click(
            fn=make_visible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_gradio_tmp_confirm_button, purge_gradio_tmp_no_button]
          )
          purge_gradio_tmp_info = gr.Textbox(label="")
          purge_gradio_tmp_confirm_button.click(
            fn=remove_gradio_tmp_dir,
            outputs=[purge_gradio_tmp_info, purge_gradio_tmp_confirm_button, purge_gradio_tmp_no_button]
          )
          purge_gradio_tmp_no_button.click(
            fn=make_invisible,
            inputs=[gr.Number(2, visible=False)],
            outputs=[purge_gradio_tmp_confirm_button, purge_gradio_tmp_no_button]
          )


### App Interface
with gr.Blocks() as app:
  gr.Markdown(value="# 베든싱어 Integrated WebUI")

  with gr.Tabs():
    with gr.TabItem("README"):
      build_readme_tab()

    with gr.TabItem("Preprocessing"):
      build_preprocessing_tab()

    if ERVC.EXISTS:
      with gr.TabItem("ervc-v2"):
        build_ervc_tab()

    if DDSP_DIFF.EXISTS:
      with gr.TabItem("DDSP-Shallow-Diffusion"):
        build_ddsp_diff_tab()

    if ESovits.EXISTS:
      with gr.TabItem("esovits"):
        gr.Markdown(value="## [Enhanced SoVITS-SVC](https://github.com/beberry-hidden-singer/enhanced-sovits-svc)")

    with gr.TabItem("Listen"):
      build_listen_tab()

    with gr.TabItem("Summary"):
      build_summary_tab()

    with gr.TabItem("Purge"):
      build_purge_tab()


if __name__ == '__main__':
  app.launch(
    server_name="0.0.0.0",
    inbrowser=False,
    server_port=args.port,
    quiet=True,
    debug=args.debug
  )
