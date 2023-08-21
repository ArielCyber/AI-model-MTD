# Neural Network MTD

## Acknowledgements


 - [Tested with PyTorch TorchVision models](https://pytorch.org/vision/0.8/models.html)
 



## Usage/Examples

**Initialize**- 

To work with the MTD class you first need to load a model to the calss, you have 2 options to do so

First way- 
Using your own way to load a PyTorch model to python, with the assumption that the model will be saved into a variable called model_to_mtd  
```python
MTD_model = MTDModel(model_to_mtd)
```

Second way- 
Loading a PyTorch model that was saved using Pickle serialization 
```python
MTD_model =  MTDModel()
MTD_model.load_model_pickle("Path/To/Model.pyth")
```

You can also save the model 

```python
MTD_model.save_model_pickle("Path/To/Model.pyth")
```

**Randomzie the model-**

Create the MTD model and lists to retrieve  to orginan model
you will need to give the function 3 files to save the data into after the randomization process
```python
MTD_model.save_mtd("Path/To/Model.pyth","Path/To/weight_map.bin","Path/To/model_map.bin")
```

You can also choose a seed for the randomization

```python
MTD_model.save_mtd("Path/To/Model.pyth","Path/To/weight_map.bin","Path/To/model_map.bin",123456789)
```

From this point on the model inside the MTD class is randomzied and probably not usable

**Retrieve  the model-**

Uses the MTD model and lists to retrieve  the orginan model
you will need to give the function 3 files to load the data from for the retrieve  process
```python
MTD_model.load_mtd("Path/To/Model.pyth","Path/To/weight_map.bin","Path/To/model_map.bin")
```

From this point on the model inside the model class is de-randomzied and usable

**Use the model**
You can get the model from the calss and use it however you want
```python
MTD_model.model
```
