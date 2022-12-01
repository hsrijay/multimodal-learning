# multimodal-learning
Final project on cross modal representation learning for ECE 685

Data:
Download VoxCeleb1 data from https://mm.kaist.ac.kr/datasets/voxceleb/ and run the following commands:
~~~~
git clone https://github.com/clovaai/voxceleb_trainer
python ./dataprep.py --save_path data --download --user USERNAME --password PASSWORD 
python ./dataprep.py --save_path data --extract
python ./dataprep.py --save_path data --convert
~~~~
Directory
1. train.py - train VAE on training set
2. eval.py - evaluate VAE on test set and create saved 'visualize' pkl file
3. preprocessing.ipynb - notebook to preprocess VoxCeleb1 data
4. visualize.ipynb - notebook to visualize manifolds in different subspaces, generate example data, and conduct cross-modal retrieval
5. model_component.py - VAE architecture definition
6. slurm-2721254.out - training loss log after 8 epochs
7. train.sh - shell file for training
8. 8_msvae_a.pkl and 8_msvae_v.pkl - model weights for audio and visual VAE, respectively, trained on VoxCeleb1 dataset for 8 epochs (need to train more)\

References
* https://github.com/L-YeZhu/Learning-Audio-Visual-Correlations
* https://github.com/iffsid/mmvae
* https://github.com/clovaai/voxceleb_trainer
* https://mm.kaist.ac.kr/datasets/voxceleb/
* https://github.com/albanie/mcnCrossModalEmotions
* https://github.com/v-iashin/VoxCeleb
