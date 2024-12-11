# Music Orchestrator
## Sukru Samet Dindar (sd3705), Siavash Shams (ss6928)

The notebook files in the repo are for preparation of the dataset generated for fine-tuning MusicGen Melody and generating the samples.
The .yaml files are externally added configuration files for fine-tuning the model in Audiocraft.
For fine-tuning Music-Gen Melody, Audiocraft library by Meta is used from terminal. Thus, this part doesn't include specific scripts. 
For the detailed information about how to install and use Audiocraft: https://github.com/facebookresearch/audiocraft

Two demos are provided in Samples/Final folder to compare the baseline model with our model.
The samples under Uncompleted are just for validation reasons, so they can be ommited.
The samples under Final are the samples of the fully completed model. Two different folders can be found with two different instrument sets prompted.
Reference track used for generating these samples can be found under Sample/Final folder directly.
While the folder 1 includes samples with quite frequent instruments in music datasets, folder 2 includes less frequent instruments.
