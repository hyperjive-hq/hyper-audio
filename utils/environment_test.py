import os

#print("Python Executable Used =======================")
#import sys; print(sys.executable)
#print("Current working directory ====================")
#print(os.getcwd())
#print("System Path ==================================")
#import sys; print(*sys.path, sep="\n")
print("System Variables =============================")

for key in sorted(os.environ.keys()):
    print(f"{key}={os.environ[key]}")