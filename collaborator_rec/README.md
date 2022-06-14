# Virtual Research Assistant (VRA): Collaborator Recommendation 

The ongoing projects of Virtual Research Assistant (VRA), which is a comprehensive research recommendation system providing recommendations regarding datasets, publications, grants and collaborators for scholars of interest in the population health domain.
Developed by researchers from Department of Biostats and Data Science, UTSPH.

**1. Main contributions**:

We've highlighted a novel collaborator recommendation based on two inductive GNNs, the made-from-scratch PubMed datasets thus setting an example for similar academic needs, and a demo implementation with human evaluations for practical usages.



**2. key component snapshot**:

* GraphSAGE

![model 1: GraphSAGE]('Figure 3.jpeg?raw=true')

* TGN 

![model 2: TGN]('Figure 5.jpg?raw=true')


## File organization 

This component is organized by different models used:
* tgn: utils, models, evalutions and service related modules.
* sage : same as tgn
* baseline : same as above without service modules.
* data: demo dataset that contains the first 100 items of the data we used in our experiments (github repo data limitations). Crawled MedLine database through PubMed. Feel free to switch to your own data using the format we provided.



## Getting Started


### Some simple examples 

1. processing data to required dataset

```
from prepare_data import yearly_authors
from prepare_dataset import CollabDataset
yearly_authors(authfile = args.authfile, years = args.yrs, savepath = args.save_path + 'data/') 
dataset = CollabDataset()
graph = dataset[0].to(device)
```

2. actual training, validation  & test 
```
jupyter nbconvert --to notebook --execute sage_main.ipynb
```
Or you can easily convert the notebook to a .py file and use below
```
python sage_main.py --node_options pubs 
```

## Built With

* [torch](https://pytorch.org/): one of the most commonly used deep learning library  
* [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/): pytorch built for graphs data
* [Transformers](https://huggingface.co/transformers/): pytorch library of transformers 
* [Pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [skearn](https://scikit-learn.org/stable/), and other common machine learning packages, see requirements.txt for details


## Authors

See also the list of [contributors](github.com/ashraf-yaseen/VRA) who participated in this project.

## License

This project is licensed under the MIT License 

## Acknowledgments

* Hat tips 
* Inspiration
* etc

