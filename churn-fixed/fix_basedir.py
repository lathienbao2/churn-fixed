import re

with open('app.py', 'r') as f:
    content = f.read()

old = 'BASE_DIR = os.getcwd()'
new = '''BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()'''

content = content.replace(old, new)

with open('app.py', 'w') as f:
    f.write(content)

print("Done!")
