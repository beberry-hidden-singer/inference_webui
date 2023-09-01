#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 8/30/23 2:16 PM
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import argparse
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
import shutil
import sys
import traceback

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from fairseq import checkpoint_utils

logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser('Beberry-Hidden-Singer Integrated WebUI Argparser')
argparser.add_argument('-p', '--port', type=int, default=7865, help='listen port')
argparser.add_argument('--debug', action='store_true', help='whether to log at DEBUG level')
args = argparser.parse_args()

logging.basicConfig(
  level=logging.DEBUG if args.debug else logging.INFO,
  format="[%(asctime)s][%(name)s][%(levelname)s]%(message)s",
  datefmt='%Y-%m-%d %H:%M:%S'
)

### default device, assuming only single node in all cases
DEVICE = 'cpu'
if torch.cuda.is_available():
  DEVICE = 'cuda'
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
  DEVICE = 'mps'
logger.info("Using Device: %s", DEVICE)
DEVICE = torch.device(DEVICE)


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

# temp folder which may be deleted upon program exit
TMP_DIR = os.path.join(HOME_DIR, 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

DATA_HOME = os.path.join(HOME_DIR, 'DATA')
os.makedirs(DATA_HOME, exist_ok=True)
TARGET_DIR = os.path.join(DATA_HOME, 'target')
os.makedirs(TARGET_DIR, exist_ok=True)
RESULT_DIR = os.path.join(DATA_HOME, 'result')
os.makedirs(RESULT_DIR, exist_ok=True)


### Gradio App utils
def refresh_choices(dir_fpath, target_ext='.wav', include_full_path=False):
  choices = []
  if os.path.exists(dir_fpath):
    for name in os.listdir(dir_fpath):
      if name.endswith(target_ext):
        if include_full_path:
          choices.append(os.path.join(dir_fpath, name))
        else:
          choices.append(name)
  return {"choices": sorted(choices), "__type__": "update"}

def clean():
  return {"value": "", "__type__": "update"}

def remove_tmp_dir():
  gradio_tmp_dir = '/tmp/gradio'
  try:
    remove_dir(gradio_tmp_dir)
  except:
    return f"Failed to remove Gradio TMP dir at `{gradio_tmp_dir}`.\nThis is not a critical error."

def remove_dir(target_dirpath):
  shutil.rmtree(target_dirpath)
  return f"Removed `{target_dirpath}`"

def update_audio(audio_fname, use_target_dir=0):
  audio_dirpath = TARGET_DIR if use_target_dir else RESULT_DIR
  audio_fpath = os.path.join(audio_dirpath, audio_fname)
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

    try:
      _, audio_fpaths = self.preprocess(
        audio_fpath, sample_rate, output_dirpath=TARGET_DIR, do_normalize=normalize, peak=peak, loudness=loudness)

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

  def refresh(self):
    weights_update = refresh_choices(self.weights_dir, target_ext='.pth')
    input_audios_update = refresh_choices(TARGET_DIR)

    index_paths = []
    for root, dirs, files in os.walk(self.logs_dir, topdown=False):
      for name in files:
        if name.endswith(".index") and "trained" not in name:
          index_paths.append("%s/%s" % (root, name))
    index_update = {"choices": sorted(index_paths), "__type__": "update"}

    return weights_update, input_audios_update, index_update

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

      logger.info("Success. New ervc-v2 VC init")
      return {"visible": True, "maximum": n_spk, "__type__": "update"}, to_return_protect0, to_return_protect1

    except:
      logger.info('Failed to init ervc-v2 VC. %s' % traceback.format_exc())
      return {"visible": False, "__type__": "update"}, None, None


  def __call__(
          self,
          sid,
          input_audio_path,
          f0_up_key,
          f0_file,
          f0_method,
          file_index,
          # file_big_npy,
          index_rate,
          filter_radius,
          rms_mix_rate,
          protect,
  ):
    if not self.EXISTS:
      return "`ervc-v2` has not been set up correctly", (None, None)

    if not os.path.exists(input_audio_path):
      input_audio_path = os.path.join(TARGET_DIR, input_audio_path)

    if input_audio_path is None or not os.path.exists(input_audio_path):
      return "Valid audio fpath should be provided", (None, None)

    f0_up_key = int(f0_up_key)
    try:
      audio = self.load_audio(input_audio_path, 16000)
      audio_max = np.abs(audio).max() / 0.95
      if audio_max > 1:
        audio /= audio_max
      times = [0, 0, 0]

      if self.hubert is None:
        self.load_hubert()

      audio_opt = self.VC.pipeline(
        self.hubert,
        self.net_g,
        sid,
        audio,
        input_audio_path,
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
      index_info = "Using index:%s." % file_index if os.path.exists(file_index) else "Index not used."
      return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (index_info, times[0], times[1], times[2],), (self.tgt_sr, audio_opt)

    except:
      msg = f'Failed to init ervc-v2 VC.\n{traceback.format_exc()}'
      logger.info(msg)
      return msg, (None, None)


class DDSP:

  HOME = os.path.join(ROOT_DIR, 'DDSP-SVC')
  EXISTS = False

  def __init__(self):
    if os.path.exists(self.HOME):
      pass


# Enhanced Sovits-svc
class ESovits:

  HOME = os.path.join(ROOT_DIR, 'esovits')
  EXISTS = False

  def __init__(self):
    if os.path.exists(self.HOME):
      pass



### Module inits
PRP = PreliminaryPreprocessing()
ERVC = ERVC_V2()



### Build App
def build_readme_tab():
  with gr.Row():
    with open(os.path.join(HOME_DIR, 'README.md')) as f:
      readme = f.read()
    gr.Markdown(readme)


def build_preprocessing_tab():
  gr.Markdown(value="## [Preliminary Preprocessing](https://github.com/beberry-hidden-singer/preliminary_preprocessing) (for INFERENCE ONLY, as of Aug 30 2023)")
  gr.Markdown(value=f"Target destination on Remote: `{TARGET_DIR}`")
  gr.Markdown(value=f"* Project home: `{PRP.HOME}`")

  with gr.Group():
    with gr.Row():
      with gr.Column():
        audio = gr.Audio(type="filepath", label="Audio to Preprocess", interactive=True)
        prp_clear_tmp_button = gr.Button("Clear /tmp Cache")
        prp_clear_tmp_info = gr.Textbox(label="")
        prp_clear_tmp_button.click(fn=remove_tmp_dir, outputs=[prp_clear_tmp_info])

      with gr.Column():
        with gr.Row():
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
      with gr.Column():
        prp_button = gr.Button("Preprocess")
        prp_info = gr.Textbox(label="Output Information")
        prp_output = gr.Audio(label="Preprocessed Audio")

      prp_button.click(
        fn=PRP,
        inputs=[audio, sample_rate, normalize, peak, loudness],
        outputs=[prp_info, prp_output]
      )


def build_ervc_tab():
  gr.Markdown(value="## [Enhanced RVC-V2](https://github.com/beberry-hidden-singer/enhanced-RVC-v2)")
  gr.Markdown(value=f"* Project home: `{ERVC.HOME}`")

  with gr.Row():
    sid0 = gr.Dropdown(label="Inference Voice", choices=[], value='Click REFRESH to update', interactive=True)
    refresh_button = gr.Button("REFRESH", variant="primary")
    spk_item = gr.Slider(
      minimum=0,
      maximum=109,
      step=1,
      label="Singer/Speaker ID",
      value=1,
      visible=False,
      interactive=True,
    )

  with gr.Group():
    gr.Markdown(value="about +/- 12 key for gender conversion")
    with gr.Row():
      with gr.Column():
        vc_transform0 = gr.Number(label="Pitch Translation in int Semi-tones", value=0)

        input_audio = gr.Dropdown(
          label="Input Audio Path", choices=[], value='Click REFRESH to update', interactive=True)
        input_audio_playback = gr.Audio(type="filepath", interactive=True, label="Input Audio Playback", visible=False)
        input_audio.change(fn=update_audio, inputs=[input_audio, gr.Number(1, visible=False)], outputs=[input_audio_playback])

      with gr.Column():
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
        file_index2 = gr.Dropdown(
          label="Auto-detected Index List",
          choices=[],
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
        ERVC,
        [
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
        [vc_output1, vc_output2],
      )
    sid0.change(fn=ERVC.get_vc, inputs=[sid0, protect0, protect0], outputs=[spk_item, protect0, protect0],)

    with gr.Group():
      gr.Markdown("Timbre Fusion")
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


def build_listen_tab():
  with gr.Row():
    gr.Markdown(value="## Listen to your Audio")
    refresh_button = gr.Button("REFRESH", variant="primary")

  def listen_refresh_fn():
    tc = refresh_choices(TARGET_DIR, '.wav')
    rc = refresh_choices(RESULT_DIR, '.wav')
    return tc, rc

  def update_audio_and_info(audio_fname, use_target_dir=0):
    audio_update = update_audio(audio_fname, use_target_dir)
    audio = sf.SoundFile(audio_update['value'])
    return audio_update, audio.extra_info

  target_choices, result_choices = listen_refresh_fn()

  with gr.Row():
    with gr.Column():
      gr.Markdown("### Target Audio")
      target_dropdown = gr.Dropdown(
        label="Target Audio Path", choices=target_choices['choices'], value='Click REFRESH to update', interactive=True)
      target_audio = gr.Audio(type="filepath", interactive=True, label="Target Audio Playback", visible=False)
      target_output = gr.Textbox(label="Target Audio Info")
      target_dropdown.change(fn=update_audio_and_info, inputs=[target_dropdown, gr.Number(value=1, visible=False)], outputs=[target_audio, target_output])

    with gr.Column():
      gr.Markdown("### Result Audio")
      result_dropdown = gr.Dropdown(
        label="Result Audio Path", choices=result_choices['choices'], value='Click REFRESH to update', interactive=True)
      result_audio = gr.Audio(type="filepath", interactive=True, label="Result Audio Playback", visible=False)
      result_output = gr.Textbox(label="Result Audio Info")
      result_dropdown.change(fn=update_audio_and_info, inputs=[result_dropdown], outputs=[result_audio, result_output])

  refresh_button.click(fn=listen_refresh_fn, outputs=[target_dropdown, result_dropdown])


### App Interface
with gr.Blocks() as app:
  gr.Markdown(value="# 베든싱어 Integrated WebUI")

  with gr.Tabs():
    with gr.TabItem("README"):
      build_readme_tab()

    with gr.TabItem("Preprocessing"):
      build_preprocessing_tab()

    with gr.TabItem("ervc-v2"):
      build_ervc_tab()

    with gr.TabItem("DDSP-Shallow-Diffusion"):
      gr.Markdown(value="## [DDSP-4.0 & Shallow Diffusion](https://github.com/beberry-hidden-singer/DDSP-shallow-diffusion)")


    with gr.TabItem("esovits"):
      gr.Markdown(value="## [Enhanced SoVITS-SVC](https://github.com/beberry-hidden-singer/enhanced-sovits-svc)")

    with gr.TabItem("Listen"):
      build_listen_tab()


if __name__ == '__main__':
  app.launch(
    server_name="0.0.0.0",
    inbrowser=False,
    server_port=args.port,
    quiet=True,
    debug=args.debug
  )
