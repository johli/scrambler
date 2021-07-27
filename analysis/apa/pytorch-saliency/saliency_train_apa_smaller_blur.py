import torch
from sal.utils.pytorch_fixes_sequence import *
from sal.utils.pytorch_trainer import *
from sal.saliency_model_sequence import SaliencyModel, SaliencyLoss, get_black_box_fn
from sal.datasets import apa_doubledope_dataset
from sal.utils.resnet_encoder_sequence_dna import resnet50encoder
import pycat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# ---- config ----
# You can choose your own dataset and a black box classifier as long as they are compatible with the ones below.
# The training code does not need to be changed and the default values should work well for high resolution ~300x300 real-world images.
# By default we train on 224x224 resolution ImageNet images with a resnet50 black box classifier.
dts = apa_doubledope_dataset


#Load predictor model

class CNNClassifier(nn.Module) :
    
    def __init__(self, batch_size, lib_index=5, distal_pas=1., padding=0) :
        super(CNNClassifier, self).__init__()
        
        self.padding = padding
        
        lib_inp_numpy = np.zeros((batch_size, 13))
        lib_inp_numpy[:, lib_index] = 1.
        self.lib_inp = Variable(torch.FloatTensor(lib_inp_numpy).to(torch.device('cuda:0')))
        
        d_inp_numpy = np.zeros((batch_size, 1))
        d_inp_numpy[:, 0] = distal_pas
        self.d_inp = Variable(torch.FloatTensor(d_inp_numpy).to(torch.device('cuda:0')))
        
        self.conv1 = nn.Conv2d(4, 96, kernel_size=(1, 8))
        self.maxpool_1 = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(96, 128, kernel_size=(1, 6))
        
        self.fc1 = nn.Linear(in_features=94 * 128 + 1, out_features=256)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=256 + 13, out_features=1)
        
        self.batch_size = batch_size
        self.use_cuda = True if torch.cuda.is_available() else False
        
    def forward(self, x):
        
        if self.padding > 0 :
            x = x[..., :-self.padding]
        
        #x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = self.maxpool_1(x)
        x = F.relu(self.conv2(x))
        
        x = x.transpose(1, 3)
        x = x.reshape(-1, 94 * 128)
        
        x = torch.cat([x, self.d_inp], dim=1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = torch.cat([x, self.lib_inp], dim=1)
        
        #x = F.sigmoid(self.fc2(x))
        x = self.fc2(x)
        
        #Transform sigmoid logits to 2-input softmax scores
        x = torch.cat([-1 * x, x], axis=1)
        
        return x

model_pytorch = CNNClassifier(batch_size=32, lib_index=4, distal_pas=1., padding=51)
_ = model_pytorch.load_state_dict(torch.load("../../scrambler_pytorch_aparent/aparent_plasmid_iso_cut_distalpas_all_libs_no_sampleweights_sgd_pytorch.pth"))

model_func = lambda pretrained=True: model_pytorch

black_box_fn = get_black_box_fn(model_zoo_model=model_func, image_domain=None)
# ----------------

#Get datasets

train_dts = dts.get_train_dataset()
val_dts = dts.get_val_dataset()

print("len(train_dts) = " + str(len(train_dts)))
print("len(val_dts) = " + str(len(val_dts)))

#Evaluate validation prediction accuracy
val_dataloader = DataLoader(val_dts, batch_size=32, num_workers=1, pin_memory=True)

black_box_model = model_func(pretrained=True)

all_predictions = []
all_labels = []

device = torch.device('cuda:0')

black_box_model.to(device)

black_box_model.eval()

with torch.no_grad():
    for i_batch, sample_batch in enumerate(val_dataloader) :

        images, labels = sample_batch

        images = images.to(device)
        
        predictions = black_box_model(images).cpu()

        all_predictions.append(predictions)
        all_labels.append(labels)

all_predictions = torch.cat(all_predictions, axis=0)
all_labels = torch.cat(all_labels, axis=0)

accuracy = torch.sum(torch.max(all_predictions, 1)[1] == all_labels.data).float() / all_predictions.size(0)

print("Validation accuracy = " + str(accuracy))

black_box_model.train()

n_epochs_phase_1 = 0
n_epochs_phase_2 = 20

model_save_str = "pytorch_saliency_model_apa_doubledope_smaller_blur_resnet50_n_epochs_phase1_" + str(n_epochs_phase_1) + "_phase2_" + str(n_epochs_phase_2)

# Default saliency model with pretrained resnet50 feature extractor, produces saliency maps which have resolution 4 times lower than the input image.
#saliency = SaliencyModel(resnet50encoder(pretrained=True), 5, 64, 3, 64, fix_encoder=True, use_simple_activation=False, allow_selector=True)

##saliency = SaliencyModel(resnet50encoder(pretrained=False, num_classes=10), 5, 64, 3, 64, fix_encoder=False, use_simple_activation=False, allow_selector=False, num_classes=10)
saliency = SaliencyModel(resnet50encoder(pretrained_f=None, num_classes=2), 5, 64, 3, 64, fix_encoder=False, use_simple_activation=False, allow_selector=False, num_classes=2)

blurred_version_prob = 0.5
blur_kernel_size = 9#55
blur_sigma = 3#11

saliency_p = nn.DataParallel(saliency).cuda()
saliency_loss_calc = SaliencyLoss(black_box_fn, smoothness_loss_coef=0.005, num_classes=2, blurred_version_prob=blurred_version_prob, blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma) # model based saliency requires very small smoothness loss and therefore can produce very sharp masks

if n_epochs_phase_1 > 0 :
    optim_phase1 = torch_optim.Adam(saliency.selector_module.parameters(), 0.001, weight_decay=0.0001)

optim_phase2 = torch_optim.Adam(saliency.get_trainable_parameters(), 0.001, weight_decay=0.0001)

@TrainStepEvent()
@EveryNthEvent(4000)
def lr_step_phase1(s):
    print
    print GREEN_STR % 'Reducing lr by a factor of 10'
    for param_group in optim_phase1.param_groups:
        param_group['lr'] = param_group['lr'] / 10.


@ev_batch_to_images_labels
def ev_phase1(_images, _labels):
    __fakes = Variable(torch.Tensor(_images.size(0)).uniform_(0, 1).cuda()<FAKE_PROB)
    _targets = (_labels + Variable(torch.Tensor(_images.size(0)).uniform_(1, 1).cuda()).long()*__fakes.long())%2
    _is_real_label = PT(is_real_label=(_targets == _labels).long())
    _masks, _exists_logits, _ = saliency_p(_images, _targets)
    PT(exists_logits=_exists_logits)
    exists_loss = F.cross_entropy(_exists_logits, _is_real_label)
    loss = PT(loss=exists_loss)


@ev_batch_to_images_labels
def ev_phase2(_images, _labels):
    __fakes = Variable(torch.Tensor(_images.size(0)).uniform_(0, 1).cuda()<FAKE_PROB)
    _targets = PT(targets=(_labels + Variable(torch.Tensor(_images.size(0)).uniform_(1, 1).cuda()).long()*__fakes.long())%2)
    _is_real_label = PT(is_real_label=(_targets == _labels).long())
    _masks, _exists_logits, _ = saliency_p(_images, _targets)
    PT(exists_logits=_exists_logits)
    saliency_loss = saliency_loss_calc.get_loss(_images, _labels, _masks, _is_real_target=_is_real_label,  pt_store=PT)
    loss = PT(loss=saliency_loss)


@TimeEvent(period=5)
def phase2_visualise(s):
    pt = s.pt_store
    ##orig = auto_norm(pt['images'][0]) #Johannes commented out
    ##mask = auto_norm(pt['masks'][0]*255, auto_normalize=False)
    ##preserved = auto_norm(pt['preserved'][0])
    ##destroyed = auto_norm(pt['destroyed'][0])
    
    ##orig = auto_norm(pt['images'][0]) / 256.0 #Johannes Fix #Johannes commented out
    ##mask = auto_norm(pt['masks'][0]*255, auto_normalize=False) / 256.0
    ##preserved = auto_norm(pt['preserved'][0]) / 256.0
    ##destroyed = auto_norm(pt['destroyed'][0]) / 256.0
    
    orig = pt['images'][0][:, 0, :] #Johannes Fix
    mask = np.tile(pt['masks'][0][:, 0, :], (4, 1))
    preserved = pt['preserved'][0][:, 0, :]
    destroyed = pt['destroyed'][0][:, 0, :]
    
    orig = np.clip(orig, 0.0, 1.0)
    mask = np.clip(mask, 0.0, 1.0)
    preserved = np.clip(preserved, 0.0, 1.0)
    destroyed = np.clip(destroyed, 0.0, 1.0)
    
    print
    ##print 'Target (%s) = %s' % (GREEN_STR%'REAL' if pt['is_real_label'][0] else RED_STR%'FAKE!' , dts.CLASS_ID_TO_NAME[pt['targets'][0]]) #Johannes commented out
    print 'Target (%s) = %s' % (GREEN_STR%'REAL' if pt['is_real_label'][0] else RED_STR%'FAKE!' , dts.CLASSES[pt['targets'][0]]) #Johannes Fix
    final = np.concatenate((orig, mask, preserved, destroyed), axis=0)
    
    ##pycat.show(final) #Johannes commented out
    
    final = np.transpose(final, (0, 1))
    
    f = plt.figure(figsize=(8, 2)) #Johannes Fix
    
    plt.imshow(final, aspect='auto')
    
    plt.xticks([], [])
    plt.yticks([], [])
    
    plt.axis('off')
    
    plt.tight_layout()
    
    plt.savefig("saliency_train_intermediate_temp1.png", dpi=150, transparent=False)
    
    plt.close()


if n_epochs_phase_1 > 0 :
    nt_phase1 = NiceTrainer(ev_phase1, dts.get_loader(train_dts, batch_size=32), optim_phase1,
                     val_dts=dts.get_loader(val_dts, batch_size=32),
                     modules=[saliency],
                     printable_vars=['loss', 'exists_accuracy'],
                     events=[lr_step_phase1,],
                     computed_variables={'exists_accuracy': accuracy_calc_op('exists_logits', 'is_real_label')})
    FAKE_PROB = .5

    for epoch_ix in range(n_epochs_phase_1) :
        nt_phase1.train()

    print GREEN_STR % 'Finished phase 1 of training, waiting until the dataloading workers shut down...'

if n_epochs_phase_2 > 0 :
    nt_phase2 = NiceTrainer(ev_phase2, dts.get_loader(train_dts, batch_size=32), optim_phase2,
                     val_dts=dts.get_loader(val_dts, batch_size=32),
                     modules=[saliency],
                     #printable_vars=['loss', 'exists_accuracy'],
                     printable_vars=['loss'],
                     events=[phase2_visualise,],
                     #computed_variables={'exists_accuracy': accuracy_calc_op('exists_logits', 'is_real_label')})
                     computed_variables={})
    FAKE_PROB = .3 if n_epochs_phase_1 > 0 else .0

    for epoch_ix in range(n_epochs_phase_2) :
        nt_phase2.train()

saliency.save(model_save_str)
