from utils              import u_loadYaml
from feat_extract       import *
from clustering         import *

if __name__ == "__main__":

    functs  = {'train_cons'     : fext_contrastive,
                'feat_ext'      : feat_exctraction,
                'clustering'    : clustering}

    confs   = u_loadYaml('conf.yml')
    print(confs)
    functs[confs.task](confs)
    
    
    