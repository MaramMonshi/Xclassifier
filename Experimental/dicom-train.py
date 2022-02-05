from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *
import pydicom
import pandas as pd
import os
data_path = Path('/home/jupyter/data/mimic-cxr-dcm')
df = pd.read_csv(data_path/'labeles-dcm.csv')
df = df.dropna(subset=['labeles'])
df = df.rename(columns={"path": "image_path"})
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
SEED = 42
seed_everything(SEED)
os.chdir("/home/jupyter/data/mimic-cxr-dcm/")


class PILDicom2(PILBase):
    "same as PILDicom but changed pixel type to np.int32 as np.int16 cannot be handled by PIL"
    _open_args,_tensor_cls,_show_args = {},TensorDicom,TensorDicom._show_args
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        "Open a `DICOM file` from path `fn` or bytes `fn` and load it as a `PIL Image`"
        if isinstance(fn,bytes): im = Image.fromarray(dcmread2(pydicom.filebase.DicomBytesIO(fn)))
        if isinstance(fn,(Path,str)): im = Image.fromarray(dcmread2(fn).astype(np.int32))  
        im.load()
        im = im._new(im.im)
        return cls(im.convert(mode) if mode else im)

db = DataBlock(blocks=(ImageBlock(cls=PILDicom), MultiCategoryBlock),
                      get_x=ColReader('image_path'),
                      get_y=ColReader('labeles', label_delim=','),
                      splitter=RandomSplitter(),
                      item_tfms=[Resize(size= 512, resamples= (Image.NONE,0))], 
                      batch_tfms=[IntToFloatTensor(div=2**16-1), *aug_transforms()])
dl = db.dataloaders(df)
learn = cnn_learner(dl, resnet18, metrics=partial(accuracy_multi, thresh=0.5))
learn.fine_tune(10) 