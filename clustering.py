import  sklearn             as sk
import  torch
import  plotly.express      as px 

from    utils               import *


def clustering(confs):
    lconfs      = confs.clustering
    data_pt     = f'{confs.local_pt.colab if u_detect_environment()[0] else confs.local_pt.local}/data'  

    #---------------------------------------------------------------------------
    # Load data
    embs        = torch.load(f'{data_pt}/{lconfs.model_ver}/embeddings.pt', weights_only=False)
    embs_2d     = torch.load(f'{data_pt}/{lconfs.model_ver}/embeddings_2d.pt', weights_only=False)
    #---------------------------------------------------------------------------    
    # Cluster
    #clusters    = sk.cluster.KMeans(n_clusters=10).fit_predict(embs)
    clusters    = sk.cluster.SpectralClustering(n_clusters=10).fit_predict(embs)
        
    #---------------------------------------------------------------------------
    # Save clusters
    clusters_pt = f'{data_pt}/{lconfs.model}/labels.pt'
    torch.save(clusters, clusters_pt)
    print(f"Saving labels to {clusters_pt}")

    #---------------------------------------------------------------------------
    if lconfs.visualize:
        visualize_clusters(embs_2d, clusters)            

#################################################################################   
#################################################################################
def visualize_clusters(embs_2d, clusters):
    # Visualize
        # sns.set_style('whitegrid')
        # plt.figure(figsize=(8, 6))
        # sns.scatterplot(x=embs_2d[:, 0], y=embs_2d[:, 1], hue=clusters, palette=sns.color_palette('pastel6', 10),
        #                 s=40, edgecolor="gray", linewidth=.5)
        # plt.show()  

        # Crear scatterplot interactivo en Plotly
        fig = px.scatter(x=embs_2d[:, 0], y=embs_2d[:, 1], 
                        color=clusters, 
                        title="Clusters en 2D",
                        color_discrete_sequence=px.colors.qualitative.Pastel,  
                        labels={"x": "Dim 1", "y": "Dim 2"},
                        hover_data={"x": embs_2d[:, 0], "y": embs_2d[:, 1], "Cluster": clusters})

        # Ajustar tamaño de los puntos (equivalente a `s=40`)
        fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="gray")))
        fig.show()

