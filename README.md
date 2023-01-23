
<h1 align="center">
  <br>
  Generalization without Systematicity reproduction and extension with transformer-based approach
  <br>
</h1>

<h4 align="center">Reproduction of the results from the paper </h4>

<p align="center">
 <a href="#about">About the project</a> 
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a>
</p>


## About
Humans have excellent compositional skills. Once a person learns the meaning of a new verb he or she can immediately compose the verb with other words to create meaningful sentence compositions. In the paper Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks (Lake and Baroni, 2018) they test the zero-shot generalization capabilities of a variety of recurrent neural networks where the generalization requires systematic compositional skills. We will in this project try to reproduce the results presented in this paper. Moreover, we will extend this approach to using a transformer-based model to test its compositional skills.

## Key Features

* Experiment tracking with Weights and Biases
* Extensible Sequence to Sequence model and trainer
* Easy extension for more experiments 
* Hyperparameter sweep with wandb

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and Python3 installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/christian2903/ATNLP

# Go into the repository
$ cd ATNLP

# Install dependencies
$ pip3 install -r requirements.txt

# Run the experiments
$ python3 main.py
```


## License

MIT
