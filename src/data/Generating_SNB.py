# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:10:40 2019

@author: saimunikoti
"""
import numpy as np
import pandas as pd
# generating system knowledge base
 
class systemknowbase():
       
    def __init__(self):
              
        print("class Systemknowbase is invoked")
        self.positions = [0, 1, 2, 4, 5, 6, 2, 7, 7, 2, 3]
        self.analytime = 96 # logged freq 15 minute for one day   
        self.filledelem =11 # 11 defined functions on
              
    def get_sysknowbase(self): 
                      
#        js= pd.DataFrame(data = np.zeros((11,8)), columns=['M1','M2','M3','B1','H1','H2','H3','H4'],
#                         index=['p1','p2','p3','p6','p24','p26','p46','p63','p66','p78','p83'])
        
        js = []
        
        for iind in range(self.analytime):
            values = np.random.uniform(0,1,self.filledelem)
            jsarray = np.zeros((11,8))
            
            for kind in range(self.filledelem):
                jsarray[kind, self.positions[kind]] = values[kind] 
            js.append(jsarray) 
        
        return js

    def get_sysknowbasedf(self,js,sampleno):
        
#        jsdf= pd.DataFrame(data = js[sampleno], columns=['M1','M2','M3','B1','H1','H2','H3','H4'],
#                         index=['p1','p2','p3','p6','p24','p26','p46','p63','p66','p78','p83'])

        jsdf= pd.DataFrame(data = js[sampleno], columns=['M1','M2','M3','B1','H1','H2','H3','H4'],
                         index=['Treat water','Gen Elec.','Consume Water','carry water M1-M3','T/P power M2-M1','T/P power M2-M3','charge EV at M3','D/C EV M3-B1','D/C EV B1-M3','Store EV at M3','Store EV at B1'])
  
        return jsdf

    
systemobj = systemknowbase()
js = systemobj.get_sysknowbase()
jsdf = systemobj.get_sysknowbasedf(js, 10)



