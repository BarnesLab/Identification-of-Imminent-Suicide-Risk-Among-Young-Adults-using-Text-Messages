[![license](http://kowsari.net/ACMCHI_DOI.svg?maxAge=2592000)](https://doi.org/10.1145/3173574.3173987)
[![GitHub license](https://img.shields.io/badge/licence-GPL-blue.svg)](./LICENSE)



# Identification of Imminent Suicide Risk Among Young Adults using Text Messages

<p align="center">
<img src="http://kowsari.net/onewebmedia/ACMCHI.jpg" width="55%"></img> 
</p>

## Documentation
Suicide is the second leading cause of death among young adults but the challenges of preventing suicide are significant because the signs often seem invisible. Research has shown that clinicians are not able to reliably predict when someone is at greatest risk. In this paper, we describe the design, collection, and analysis of text messages from individuals with a history of suicidal thoughts and behaviors to build a model to identify periods of suicidality (i.e., suicidal ideation and non-fatal suicide attempts). By reconstructing the timeline of recent suicidal behaviors through a retrospective clinical interview, this study utilizes a prospective research design to understand if text communications can predict periods of suicidality versus depression. Identifying subtle clues in communication indicating when someone is at heightened risk of a suicide attempt may allow for more effective prevention of suicide.


## Installation ##

There are git in this repository; to clone all the needed files, please use:

    git clone --recursive https://github.com/BarnesLab/Identification-of-Imminent-Suicide-Risk-Among-Young-Adults-using-Text-Messages.git
     
     
The primary requirements for this package are Python 3 with Tensorflow. The requirements.txt file contains a listing of the required Python packages; to install all requirements, run the following:
    
    pip -r install requirements.txt
    
Or

    pip3  install -r requirements.txt

Or:

    conda install --file requirements.txt
        
If the above command does not work, use the following:

    sudo -H pip  install -r requirements.txt
    

## General: ##

- Python 3.5 or later see [Instruction Documents](https://www.python.org/)

- TensorFlow see [Instruction Documents](https://www.tensorflow.org/install/install_linux).

- scikit-learn see [Instruction Documents](http://scikit-learn.org/stable/install.html)

- Keras see [Instruction Documents](https://keras.io/)

- scipy see [Instruction Documents](https://www.scipy.org/install.html)

- GPU (if you want to run on GPU):

  * CUDA® Toolkit 8.0. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cuda-toolkit). 

  * The [NVIDIA drivers associated with CUDA Toolkit 8.0](http://www.nvidia.com/Download/index.aspx).

  * cuDNN v6. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cudnn). 

  * GPU card with CUDA Compute Capability 3.0 or higher.

  * The libcupti-dev library,


## Error and Comments: ##

Send an email to [kk7nc@virginia.edu](mailto:kk7nc@virginia.edu)


## Citation: ##

    @inproceedings{nobles2018identification,
      title={Identification of Imminent Suicide Risk Among Young Adults using Text Messages},
      author={Nobles, Alicia L. and Glenn, Jeffrey J. and Kowsari, Kamran and Teachman, Bethany A. and Barnes, Laura E.},
      booktitle={Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems},
      year={2018},
      organization={ACM},
      doi={10.1145/3173574.3173987}
    }


