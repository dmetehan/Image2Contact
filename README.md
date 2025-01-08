# Image2Contact

Image2Contact is the multimodal method proposed in the paper "Decoding Contact: Automatic Estimation of Contact Signatures in Parent-Infant Free Play Interactions".

Our unimodal approach can be reached via [Pose2Contact](https://github.com/dmetehan/Pose2Contact)

To annotate your own data with contact signatures you can use our annotation tool [HumanContactAnnotator](https://github.com/dmetehan/HumanContactAnnotator)

# Usage:

- Download the model weights from the following link and extract it in the repository: https://drive.google.com/file/d/1JR0SPl3nWsTtlYyjjJd9bNs2iKC5DYYz/view?usp=sharing
- Download the data from the following link and extract it in the repository: https://drive.google.com/file/d/1Q6Y_Izb5IBSU7MTjkGIjkSyaRXyV3QXa/view?usp=drive_link
- Request YOUth PCI dataset 10 month old wave: https://www.uu.nl/en/research/youth-cohort-study/request-youth-data
- Extract each 5 second frames, crops, joint_hmaps, bodyparts and store in a folder (<path_to_YOUth10mSignatures>) 
- Run YOUth_test.py with the following parameters: "<path_to_YOUth10mSignatures> configs/backbones/bb0_config.yaml exp/YOUth_cross --test_set"

## TODO:

- [ ] Uploading Model Weights
- [ ] Demo script
- [ ] Installation Instructions

## Citation

To cite our work:
```
@inproceedings{10.1145/3678957.3685719,
author = {Doyran, Metehan and Salah, Albert Ali and Poppe, Ronald},
title = {Decoding Contact: Automatic Estimation of Contact Signatures in Parent-Infant Free Play Interactions},
year = {2024},
isbn = {9798400704628},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3678957.3685719},
doi = {10.1145/3678957.3685719},
booktitle = {Proceedings of the 26th International Conference on Multimodal Interaction},
pages = {38–46},
location = {San Jose, Costa Rica},
series = {ICMI '24}
}
```
