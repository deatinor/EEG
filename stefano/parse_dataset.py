import os
from utilities import * 

import matplotlib.pyplot as plt
from collections import namedtuple

class Dataset(os.PathLike):
    def __init__(self,dataset,classes=3,
            base_directory='/cvlabdata1/cvlab/datasets_stefano/',
            format_images='png'):
        
        # Compute self directory
        datasets=next(os.walk(base_directory))[1]
        dataset_folder=find_more_similar(dataset.lower(),datasets)+'/'
        self.directory=base_directory+dataset_folder+'/'
        print("Reading data from",self.directory)
        
        directories=next(os.walk(self.directory))[1]
        for i in directories:
            if i.startswith('.'):
                continue
            dataset=Dataset(i,base_directory=self.directory)
            setattr(self,i,dataset)


        
        
    def folder(self,*args):
        output_folder=self.directory
        
        for i in args:
            directories=next(os.walk(output_folder))[1]
            new_folder=find_more_similar(str(i).lower(),directories)
            output_folder+=new_folder+'/'

        return output_folder


    def file(self,*args):
        output_folder=self.directory
        
        for i in args[:-1]:
            directories=next(os.walk(output_folder))[1]
            new_folder=find_more_similar(str(i).lower(),directories)
            output_folder+=new_folder+'/'

        files=os.listdir(output_folder)
        f=find_more_similar(str(args[-1]),files)
        return output_folder+f

    def __str__(self):
        return self.directory

    def __repr__(self):
        return str(self)

    def __add__(self,x):
        return str(self)+x

    def __radd__(self,x):
        return x+str(self)

    def __fspath__(self):
        return str(self)

    def __iter__(self):
        for i in self.directory:
            yield i
    
