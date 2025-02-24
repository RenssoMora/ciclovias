from utils              import u_loadYaml
from feat_extract       import *

if __name__ == "__main__":
    confs = u_loadYaml('conf.yml')
    print(confs)
    Fext_contrastive(confs)
    
    