# Virtual Research Assistant (VRA): Dataset Recommendation 

The ongoing projects of Virtual Research Assistant (VRA), which is a comprehensive research recommendation system providing recommendations regarding datasets, publications, grants and collaborators for scholars of interest in the population health domain.
Developed by researchers from Department of Biostats and Data Science, UTSPH.

**1. Main contributions**:


This component consisits of 3 sub-projects
1. data2lit: recommending literature to datasets by exploring a variety of vector representations from IR techniques to non-contextual embeddings to contextual embeddings. See [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8378599/)
2. datarec: actual data recommendation project. Using BERT-based model.
3. sensitivity analysis: study how results in 2. differ with different training data imbalance ratio. See [paper](https://journals.flvc.org/FLAIRS/issue/view/6020)


**2. key component snapshot**:


All 3 sub-projects had similar process. The whole system looks like below:

![system]('1.system.png?raw=true')


and main model usage:

![model]('2.bertusage.png?raw=true')



## File organization 

This component is organized as below:
* data2lit: preliminary subproject. Early stage code, kinda messy and un-modularized. Each methods has its own master file, utils related in the utils folder, embedding models related in the emb folder, classification formulation experiments in the clf folder.
* datasetrec: actual dataset recommendation. Main execution files and its model, dataloader, utils and evaluation modules
* sensitivity: sensitivity analyis on class imbalance to dataset recommendation results. Main execution files and its model, dataloader, utils and evaluation modules, as well as data collection module in the separate folder in dataPreprocess
* scrapping: preliminary data collection 
* data: very small sample data. Data repo meta information, publication meta information, existing pairs from the meta and false pairs created.



## Getting Started


### Some simple examples 

Directly using the code from the repo
```
jupyter nbconvert --to notebook --execute main.ipynb
```
Or the .py file
```
python main.py --load_pretrained False
```

## Built With

* [torch](https://pytorch.org/): one of the most commonly used deep learning library  
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

