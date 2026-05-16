import re

with open('/home/zhangshibo24s/cell_flow/cellflow/data/_datamanager.py', 'r') as f:
    content = f.read()

# fix the categorical multiply error
if "return np.ones((1, 1)) * arr" in content:
    content = content.replace("return np.ones((1, 1)) * arr", 
"""# fix categorical array multiplication
    if arr.dtype.kind in {'U', 'S', 'O'} or str(arr.dtype) == 'category':
        return np.tile(arr, (1, 1))
    return np.ones((1, 1)) * arr""")
    
with open('/home/zhangshibo24s/cell_flow/cellflow/data/_datamanager.py', 'w') as f:
    f.write(content)
print('Fixed DataManager categorical logic.')
