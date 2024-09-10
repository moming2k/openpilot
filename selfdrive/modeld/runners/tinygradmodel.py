import os
import sys
from pathlib import Path
OPENPILOT_ROOT = Path(__file__).parent.parent.parent.parent
if not (OPENPILOT_ROOT / 'tinygrad_repo').exists():
  OPENPILOT_ROOT = OPENPILOT_ROOT.parent  # This takes us from openpilot/openpilot to openpilot but it's probably wrong
TINYGRAD_REPO_PATH = OPENPILOT_ROOT / 'tinygrad_repo'
assert TINYGRAD_REPO_PATH.exists()
assert (TINYGRAD_REPO_PATH / 'extra').exists()
sys.path.append(str(TINYGRAD_REPO_PATH))

os.environ['GPU'] = '1'

import onnx
import numpy as np

from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from extra.onnx import get_run_onnx
from openpilot.selfdrive.modeld.runners.runmodel_pyx import RunModel

ONNX_TYPES_TO_NP_TYPES: dict[int, np.dtype] = {
  i: onnx.helper.tensor_dtype_to_np_dtype(i)
    for dtype, i in onnx.TensorProto.DataType.items()
    if dtype in ['FLOAT', 'FLOAT16', 'INT64', 'INT32', 'UINT8']
}

@TinyJit
def model_exec(run_onnx, **inputs):
  return next(iter(run_onnx(inputs).values())).cast(dtypes.float32).realize()


class TinygradModel(RunModel):
  def __init__(self, path, output, runtime, use_tf8, cl_context):
    self.inputs = {}
    self.output = output
    self.use_tf8 = use_tf8

    Tensor.manual_seed(1337)
    Tensor.no_grad = True

    onnx_model = onnx.load(path)
    self.run_onnx = get_run_onnx(onnx_model)
    self.input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
    self.input_dtypes = {inp.name: ONNX_TYPES_TO_NP_TYPES[inp.type.tensor_type.elem_type] for inp in onnx_model.graph.input}

  def addInput(self, name, buffer):
    assert name in self.input_shapes
    self.inputs[name] = buffer

  def setInputBuffer(self, name, buffer):
    assert name in self.inputs
    self.inputs[name] = buffer

  def getCLBuffer(self, name):
    return None

  def execute(self):
    inputs = {k: (v.view(np.uint8) / 255. if self.use_tf8 and k == 'input_img' else v) for k,v in self.inputs.items()}
    inputs = {k: v.reshape(self.input_shapes[k]).astype(self.input_dtypes[k]) for k,v in inputs.items()}
    inputs = {k: Tensor(v) for k,v in inputs.items()}
    outputs = model_exec(self.run_onnx, **inputs)
    self.output[:] = outputs.numpy()
    return self.output


if __name__ == "__main__":
  import pickle
  from tqdm import trange
  from openpilot.selfdrive.modeld.runners import Runtime
  from openpilot.selfdrive.modeld.constants import ModelConstants
  from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext

  MODEL_PATH = Path(__file__).parent.parent / 'models/supercombo.onnx'
  METADATA_PATH = Path(__file__).parent.parent / 'models/supercombo_metadata.pkl'
  with open(METADATA_PATH, 'rb') as f:
    model_metadata = pickle.load(f)

  net_output_size = model_metadata['output_shapes']['outputs'][1]
  output = np.zeros(net_output_size, dtype=np.float32)

  model = TinygradModel(MODEL_PATH, output, Runtime.CPU, False, CLContext())

  inputs = {
    'input_imgs': np.zeros(128 * 256 * 12, dtype=np.uint8),
    'big_input_imgs': np.zeros(128 * 256 * 12, dtype=np.uint8),
    'desire': np.zeros(ModelConstants.DESIRE_LEN * (ModelConstants.HISTORY_BUFFER_LEN+1), dtype=np.float32),
    'traffic_convention': np.zeros(ModelConstants.TRAFFIC_CONVENTION_LEN, dtype=np.float32),
    'lateral_control_params': np.zeros(ModelConstants.LATERAL_CONTROL_PARAMS_LEN, dtype=np.float32),
    'prev_desired_curv': np.zeros(ModelConstants.PREV_DESIRED_CURV_LEN * (ModelConstants.HISTORY_BUFFER_LEN+1), dtype=np.float32),
    'features_buffer': np.zeros(ModelConstants.HISTORY_BUFFER_LEN * ModelConstants.FEATURE_LEN, dtype=np.float32),
  }

  for k,v in inputs.items():
    model.addInput(k, v)

  for _ in trange(100):
    model.execute()
