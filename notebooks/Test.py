
# coding: utf-8

# # Check Swadesh quality

# In[36]:

import numpy as np
import os


# In[39]:

dirname = '/home/eszti/data/panlex_swadesh/swadesh110'
not_found_list = np.zeros(110)
no_one_word_list = np.zeros(110)

lang_codes = set()

for fn in os.listdir(dirname):
    if fn.endswith(".txt"):
        code = fn.split('-')[0]
        print(code)
        if code in lang_codes:
            print('{0} is already in the set'.format(code))
            continue
        lang_codes.add(code)       
        with open(os.path.join(dirname, fn)) as f:
            i = 0
            for line in f:
                fields = line.strip().decode('utf-8').split('\t')
                if len(fields) == 1 and fields[0] == '':
                    not_found_list[i] += 1
                else:
                    found = False
                    for w in fields:
                        if ' ' not in w:
                            found = True
                            break
                    if not found:
#                         print(fields)
                        no_one_word_list[i] += 1
                i += 1


# This list shows in case of how many languages there was found no entry at each position

# In[40]:

not_found_list


# This list shows in case of how many languages there was no entry without a space in it

# In[41]:

no_one_word_list


# In[ ]:



