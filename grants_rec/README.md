# Virtual Research Assistant (VRA): Grant Recommendation 

The ongoing projects of Virtual Research Assistant (VRA), which is a comprehensive research recommendation system providing recommendations regarding datasets, publications, grants and collaborators for scholars of interest in the population health domain.
Developed by researchers from Department of Biostats and Data Science, UTSPH.

**1. Main contributions**:

We proposed a grant recommendation system for National Institute of Health (NIH) grants using researchers’ publications. A researcher’s areas of expertise are identified by a clustering method: Dirichlet Process Mixture Models (DPMM) on the researcher’s publications.  The relevant grants opportunities were then recommended using a BERT-based recommender.



**2. key component snapshot**:

* The training stage

![training](Fig-5.png?raw=true)

* The service stage

![service](Fig-6.png?raw=true)


## File organization 

This component is organized as below:
* proposed: the main file together with data processing, utils, models modules
* baseline : same as above for all baselines: tf-idf, bm25, word2ec, and naive Bayes with tfidf
* service: extension to proposed, for the service stage after proposed method is trained: main file together with CV, data processing & utils modules 
* data: demo dataset that contains the first 100 items of the data we used in our experiments (github repo data limitations). Feel free to switch to your own data using the format we provided.
* eval_metrics.py: evaluation module shared by all methods.



## Getting Started


### Some simple examples 

Directly using the code from the repo
```
jupyter nbconvert --to notebook --execute main_grant.ipynb
```
Or you can easily convert the notebook to a .py file and use the .py file as you do on the command line 
```
$ jupyter nbconvert --to main_grant.py main_grant.ipynb
python main_grant.py --train_epochs 10
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

