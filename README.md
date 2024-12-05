# 570Submission
This contains the github for an implementation for Efiificient Equivariant Subsampling, ES4-Net. The ES4-Net is compared to several baseline models. After running the requirement.txt setup, use the command line interface to select which models to train and demo.
### Command line instructions
python model/main.py <modelName> <trainingOptions>

Model Name:
* "Vanilla" selects a vanilla CNN 
* "Basic-Net" selects a baseline equivariant network
* "E4-Net" selects the E4-Net, upon which this worked is based
* "ES4-Net" selects the model this work presents

Training Options
* -e selects to train the model on the c4 representation of the mnist dataset
* -ne selects to train the model on the original mnist datset
