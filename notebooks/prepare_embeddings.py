
# coding: utf-8

# # Prepare embeddings

# FB - SIL codes mapping

# In[15]:

import json


# In[21]:

file = '/home/eszti/projects/dipterv/univ_embedding/res/temp.txt'
fb2sil = dict()
sil2fb = dict()

with open(file) as f:
    lines = f.readlines()
    for line in lines:
        if 'SIL' in line:
            sil =  line.split('/')[1]
#             print 'SIL: ' + sil
        if 'FB_' in line:
            fb = line.replace('FB_', '').replace('\n', '')
#             print 'FB: ' + fb
            if fb in fb2sil:
                print fb
                print fb2sil[fb]
                print sil
            fb2sil[fb] = sil
            sil2fb[sil] = fb
print len(fb2sil)
print len(sil2fb)


# Save facebook-sil code mapping

# In[22]:

fbcodes_fn = '/home/eszti/projects/dipterv/univ_embedding/res/fb2silcodes.json'
with open(fbcodes_fn, 'w') as f:
    json.dump(fb2sil, f)


# In[30]:

fn = '/home/eszti/projects/dipterv/univ_embedding/res/wikicodes.json'
with open(fn, 'w') as f:
    json.dump(wikicodes, f)


# In[ ]:



