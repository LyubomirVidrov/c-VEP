# c-VEP

### Setup 
1. Use Python `3.11.15`.
1. Install all the requirements as follows:
```bash
   pip install braindecode==1.5.0 moabb==1.4.3 pyntbci==1.8.3
```
3. Clone MOABB locally because we need to modify the preprocessing step to match rCCA. 
4. Open:
`moabb/paradigms/base.py`
5. Inside the `BaseProcessing` class. Ensure you have the following:
   ```python
    RawToEpochs(
     event_id=self.used_events(dataset),
     tmin=bmin - 0.5,
     tmax=bmax,
     baseline=baseline,
     channels=self.channels,
     interpolate_missing_channels=self.interpolate_missing_channels,
     return_all_modalities=dataset.return_all_modalities,
   )

   if self.resample is not None:
     steps.append(("resample", get_resample_pipeline(self.resample)))
   
   # crop after the resampling to account for the artifacts introduced by resampling 
   if bmin  - 0.5 < tmin or bmax > tmax:
     steps.append(("crop", get_crop_pipeline(tmin=tmin, tmax=tmax)))
   ```
### Run

The following steps should be performed in order for both the linear and nonlinear approaches:

1. Run the preprocessing scripts.
2. Run `training.py` to determine the optimal parameter combination. Use these parameters in the subsequent steps.
3. Run `learning_curve.py`.
4. Run `decoding_curve.py`.
5. Run `main.py` to generate and display the figures.

### References
Dataset: 
```bibtex
@dataset{jordy_thielen_2023_from_full_calibratio,
  author = {Thielen, J. (Jordy) and Pieter Marsman and Jason Farquhar and Desain, P.W.M. (Peter)},
  title = {{From full calibration to zero training for a code-modulated visual evoked potentials brain computer interface}},
  year = 2023,
  publisher = {Radboud University},
  version = 3,
  doi = {10.34973/9txv-z787},
  url = {https://doi.org/10.34973/9txv-z787}
}
```

Replicated code + PyntBCI:
```bibtex
@article{thielen2021full,
  title={From full calibration to zero training for a code-modulated visual evoked potentials for brain--computer interface},
  author={Thielen, Jordy and Marsman, Pieter and Farquhar, Jason and Desain, Peter},
  journal={Journal of Neural Engineering},
  volume={18},
  number={5},
  pages={056007},
  year={2021},
  publisher={IOP Publishing}
}
```

MOABB:
```bibtex
@software{Aristimunha_Mother_of_all_2023,
author = {Aristimunha, Bruno and Carrara, Igor and Guetschel, Pierre and Sedlar, Sara and Rodrigues, Pedro and Sosulski, Jan and Narayanan, Divyesh and Bjareholt, Erik and Quentin, Barthelemy and Schirrmeister, Robin Tibor and Kalunga, Emmanuel and Darmet, Ludovic and Gregoire, Cattan and Abdul Hussain, Ali and Gatti, Ramiro and Goncharenko, Vladislav and Thielen, Jordy and Moreau, Thomas and Roy, Yannick and Jayaram, Vinay and Barachant, Alexandre and Chevallier, Sylvain},
doi = {10.5281/zenodo.10034223},
title = {{Mother of all BCI Benchmarks}},
url = {https://github.com/NeuroTechX/moabb},
version = {1.0.0},
year = {2023}
}
```

Braindecode:
```bibtex
@misc{braindecode,
  author = {Aristimunha, Bruno and
            Guetschel, Pierre and
            Wimpff, Martin and
            Gemein, Lukas and
            Rommel, Cedric and
            Banville, Hubert and
            Sliwowski, Maciej and
            Wilson, Daniel and
            Brandt, Simon and
            Gnassounou, Théo and
            Paillard, Joseph and
            {Junqueira Lopes}, Bruna and
            Sedlar, Sara and
            Moreau, Thomas and
            Chevallier, Sylvain and
            Gramfort, Alexandre and
            Schirrmeister, Robin Tibor},
  title = {Braindecode: toolbox for decoding raw electrophysiological brain data
           with deep learning models},
  howpublished = {Zenodo},
  url = {https://github.com/braindecode/braindecode},
  doi = {10.5281/zenodo.17699192},
  license = {BSD-3-Clause},
}
```
