![Scrambler Logo](https://github.com/johli/scrambler/blob/master/resources/scrambler_logo.png?raw=true)

# Scrambler Neural Networks
Code for training Scrambler networks, an interpretation method for sequence-predictive models based on deep generative masking. The Scrambler learns to predict maximal-entropy PSSMs for a given input sequence such that downstream predictions are reconstructed (the "inclusion" objective). Alternatively, the Scrambler can be trained to output minimal-entropy PSSMs such that downstream predictions are distorted (the "occlusion" objective).

Scramblers were presented in a MLCB 2020* conference paper, "[Efficient inference of nonlinear feature attributions with Scrambling Neural Networks](https://drive.google.com/file/d/142tmyEMLUSsV-IEkN-NFcEUd7-LFwaAF/view?usp=sharing)".

*2nd Conference on Machine Learning in Computational Biology, (MLCB 2020), Online.

Contact *jlinder2 (at) cs.washington.edu* for any questions about the code.

#### Features
- Efficient interpretation of sequence-predictive neural networks.
- High-capacity interpreter based on ResNets.
- Find multiple salient feature sets with mask dropout.
- Separate maximally enhancing and repressive features.
- Fine-tune interpretations with per-example optimization.
- Supports multiple-input predictor architectures.

### Installation
Install by cloning or forking the [github repository](https://github.com/johli/scrambler.git):
```sh
git clone https://github.com/johli/scrambler.git
cd scrambler
python setup.py install
```

#### Required Packages
- Tensorflow == 1.13.1
- Keras == 2.2.4
- Scipy >= 1.2.1
- Numpy >= 1.16.2

### Analysis Notebooks 
The sub-folder **analysis/** contains all the code used to produce the results of the paper.

### Example Notebooks
The sub-folder **examples/** contains a number of light-weight examples showing the basic usage of the Scrambler package functionality. The examples are listed below.

#### Images
Interpretating predictors for images.

[Notebook 1: Interpreting MNIST Images](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/image/scrambler_mnist_example.ipynb)<br/>

#### RNA
Interpretating predictors for RNA-regulatory biology.

[Notebook 2a: Interpreting APA Sequences](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/dna/scrambler_apa_example.ipynb)<br/>
[Notebook 2b: Interpreting APA Sequences (Custom Loss)](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/dna/scrambler_apa_example_custom_loss.ipynb)<br/>
[Notebook 3a: Interpreting 5' UTR Sequences](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/dna/scrambler_optimus5_example.ipynb)<br/>
[Notebook 3b: Optimizing individual 5' UTR Interpretations](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/dna/scrambler_optimus5_from_scratch_example.ipynb)<br/>
[Notebook 3c: Fine-tuning pre-trained 5' UTR Interpretations](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/dna/scrambler_optimus5_finetuning_example.ipynb)<br/>

#### Protein
Interpretating predictors for proteins.

[Notebook 4a: Interpreting Protein-protein Interactions (inclusion)](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/protein/scrambler_ppi_example_inclusion.ipynb)<br/>
[Notebook 4b: Interpreting Protein-protein Interactions (occlusion)](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/protein/scrambler_ppi_example_label.ipynb)<br/>
[Notebook 5a: Interpreting Hallucinated Protein Structures (no MSA)](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/protein/scrambler_rosetta_example_no_msa.ipynb)<br/>
[Notebook 5b: Interpreting Natural Protein Structures (with MSA)](https://nbviewer.jupyter.org/github/johli/scrambler/blob/master/examples/protein/scrambler_rosetta_example_with_msa.ipynb)<br/>

### Scrambler Training GIFs
The following GIFs illustrate how the Scrambler network interpretations converge on a few select input examples during training.

**WARNING:** The following GIFs contain flickering pixels/colors. Do not look at them if you are sensitive to such images.

#### Alternative Polyadenylation
The following GIF depicts a Scrambler trained to reconstruct APA isoform predictions.

![APA GIF](https://github.com/johli/scrambler/blob/master/resources/apa_inclusion_scrambler_smooth_target_bits_025.gif?raw=true)

#### 5' UTR Translation Efficiency
The following GIF depicts a Scrambler trained to reconstruct 5' UTR translation efficiency predictions.

![UTR5 GIF](https://github.com/johli/scrambler/blob/master/resources/optimus5_inclusion_scrambler_bits_0125.gif?raw=true)

#### Protein-Protein Interactions
The following GIF depicts a Scrambler trained to distort protein interactions predictions (siamese occlusion). Red letters correspond to designed hydrogen bond network positions. The following GIF displays the same interpretation but projected onto the 3D structure of the complex.

![Protein GIF](https://github.com/johli/scrambler/blob/master/resources/ppi_occlusion_scrambler_bits_24.gif?raw=true)

![Protein GIF](https://github.com/johli/scrambler/blob/master/resources/ppi_occlusion_scrambler_bits_24_3d.gif?raw=true)
