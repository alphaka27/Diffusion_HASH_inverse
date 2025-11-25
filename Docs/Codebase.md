# Diffusion Hash Codebase

## Main entry point 
- hash_main.py

## Hash Implementaion
- SHA256  
Input: Byte  
Output: Byte

- MD5  
Input: Byte  
Output: Byte

## Random Generator
- Charactre Generator  
Input: None  
Output: Bytes (w. Unicode normalize & UTF-8 encoding)

- Bytes Generator  
Input: None
Output: Bytes (In big endian)

## Utility calling
- Random Generator  
    Calling
    - FileIO(class).file_io(func).file_write(func)  

    Called by: hash_main.py  

- Main Entry point  
    Calling
    - FileIO _(class)_ .file_io _(func)_ .file_write _(func)_  
    - OutputFormat _(class)_
    - Hash Implementatioin _(category)_
    - Random Generator _(category)_

### Function IO types
struct (Python Internal function)  
- pack  
    Input: String, Any  
    Output: Bytes

