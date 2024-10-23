import enum
import os
import data
import models

class Resolution(enum.Enum):
  """Common resolutions defined as (H, W)"""
  SD = (480, 640)
  HD = (720, 1280)
  UHD = (2160, 3840)
  SQUARE = (512, 512)

class Paths(enum.Enum):
  ASSETS = os.path.join(
    os.path.dirname(data.__file__), 
    'assets'
  )
  MJ_MODELS = os.path.join(
    os.path.dirname(data.__file__), 
    'mj_models'
  )
  SCENES = os.path.join(
    os.path.dirname(data.__file__), 
    'scenes'
  )
  MODELS = os.path.join(
    os.path.dirname(models.__file__)
  )
