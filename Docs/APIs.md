# Function Call structure

Execute hash_main.py (Main EntryPoint)  
&nbsp;&nbsp;&nbsp;&nbsp; $\rightarrow$ Call each hash algorithm's main function

## Hash algorithm's Structure
__Implement using class syntax__  
### Input & Output
__Input__: Bytes  
__Output__: Bytes | String

Class name is must be __UPPER CASE__  
```__init__``` function  
```python
def __init__(self, is_verbose = True, output_format = None)
```


__Hash algorithm's main function__
```python
def digest(self, message = None, message_len = -1)
```

## Random Character/Bit Generator
__Implement using class syntax__  
### Input & Output
__Input__: None  
__Output__: Bytes
