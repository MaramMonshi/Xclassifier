from fastai.vision.all import *
from timm import create_model
from fastai.vision.learner import _update_first_layer
from fastai.distributed import *
import os
from sklearn.model_selection import train_test_split
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
os.chdir("/home/jupyter/data/mimic-cxr-jpg/")
torch.cuda.empty_cache()
import gc
gc.collect()
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/home/jupyter/data/mimic-cxr-jpg/mimiccxr.csv')
df = df.dropna(subset=['labels'])
df_train, df_test = train_test_split(df, test_size=0.1, random_state=SEED, shuffle=True)

bs = 64
epoch = 30
metrics=[accuracy_multi,  
         RocAucMulti(),
         PrecisionMulti(),
         RecallMulti(),     
         F1ScoreMulti()]
item_tfms=Resize(224, method='squish', pad_mode='zeros', resamples=(2, 0))
batch_tfms=[*aug_transforms(mult=1.0, do_flip=False, flip_vert=False, 
                            max_rotate=20.0, max_zoom=1.2, max_lighting=0.3, max_warp=0.2, 
                            p_affine=0.75, p_lighting=0.75, 
                            xtra_tfms=None, size=None, mode='bilinear', pad_mode='reflection', 
                            align_corners=True, batch=False, min_scale=1.0),
                            Normalize.from_stats(*imagenet_stats)]

dl = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
              get_x=ColReader('path'),
              get_y=ColReader('labels', label_delim=','),
              splitter=RandomSplitter(valid_pct=0.1, seed = SEED),
              item_tfms=item_tfms,
              batch_tfms=batch_tfms
).dataloaders(df_train, bs=bs)

def create_body(arch, n_in=3, pretrained=True, cut=None):
    "Cut off the body of a typically pretrained `arch` as determined by `cut`"
    model = arch(pretrained=pretrained)
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if   isinstance(cut, int):      return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or a function")

def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")

body = create_timm_body('densenetblur121d', pretrained=True)
nf = num_features_model(body)
head = create_head(nf, dl.c, concat_pool=True)
net = nn.Sequential(body, head)

learn = Learner(dl, net, metrics=metrics)

test_dl = dl.test_dl(df_test, bs=bs, with_labels=True)

with learn.distrib_ctx(sync_bn=False): learn.fine_tune(epoch); print (learn.validate(dl=test_dl))   




