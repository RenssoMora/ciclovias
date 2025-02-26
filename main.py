from utils              import u_loadYaml
from feat_extract       import *
from clustering         import *

if __name__ == "__main__":

    functs  = {'train_contrastive'      : fext_contrastive,
                'feat_extraction'       : feat_extraction,
                'clustering'            : clustering,
                'val_clusters'          : val_clusters
                }

    confs   = u_loadYaml('conf.yml')
    print(confs)
    functs[confs.task](confs)
    
    
    