import  sklearn             as sk
import  torch
import  plotly.express      as px 
import  pandas              as pd
import  matplotlib.image    as mpimg
import  matplotlib.pyplot   as plt

from    tqdm                import tqdm
from    utils               import *
from    scipy.spatial.distance  import cdist




def clustering(confs):
    lconfs      = confs.clustering
    data_pt     = f'{confs.local_pt.colab if u_detect_environment()[0] else confs.local_pt.local}/data'  

    #---------------------------------------------------------------------------
    # Load data
    embs        = torch.load(f'{data_pt}/{lconfs.model_ver}/embeddings.pt', weights_only=False)
    embs_2d     = torch.load(f'{data_pt}/{lconfs.model_ver}/embeddings_2d.pt', weights_only=False)
    #---------------------------------------------------------------------------    
    # Cluster
    if lconfs.method == 'kmeans':
        wcss = []
        range_ = range(2, 30, 3)
        for i in tqdm(range_):
            wcss.append(sk.cluster.KMeans(n_clusters=i, n_init = 10).fit(embs).inertia_)

        plt.plot(range_, wcss)
        plt.title('Elbow Method')   
        plt.xlabel('Number of clusters')
        plt.grid(alpha=0.7)
        plt.ylabel('WCSS')
        plt.savefig(f'{data_pt}/{lconfs.model_ver}/elbow_method.png')
        plt.show()

        #clusters    = sk.cluster.KMeans(n_clusters=lconfs.n_clusters).fit(embs)
    elif lconfs.method == 'spectral':
        clusters    = sk.cluster.SpectralClustering(n_clusters=lconfs.n_clusters, 
                                                    n_neighbors=40,
                                                    affinity='nearest_neighbors',
                                                    eigen_solver='arpack').fit_predict(embs)
        
    #---------------------------------------------------------------------------
    # Save clusters
    clusters_pt = f'{data_pt}/{lconfs.model_ver}/labels.json'
    u_save2json(clusters_pt, clusters.tolist())
    u_save2Yaml(f'{data_pt}/{lconfs.model_ver}/clustering_history.yml', 
                u_class2dict(lconfs))

    #---------------------------------------------------------------------------
    if lconfs.visualize:
        visualize_clusters(embs_2d, clusters, f'{data_pt}/{lconfs.model_ver}', True)            

#################################################################################   
#################################################################################
def visualize_clusters(embs_2d, clusters, base_out_pt, write=False, ids = None):
    # Visualize
        # sns.set_style('whitegrid')
        # plt.figure(figsize=(8, 6))
        # sns.scatterplot(x=embs_2d[:, 0], y=embs_2d[:, 1], hue=clusters, palette=sns.color_palette('pastel6', 10),
        #                 s=40, edgecolor="gray", linewidth=.5)
        # plt.show()  

        # Crear scatterplot interactivo en Plotly
        if ids is None:
            fig = px.scatter(x=embs_2d[:, 0], 
                         y=embs_2d[:, 1], 
                        color=clusters, 
                        title="Clusters en 2D",
                        color_discrete_sequence=px.colors.qualitative.Pastel,  
                        labels={"x": "Dim 1", "y": "Dim 2"},
                        hover_data={"x": embs_2d[:, 0], "y": embs_2d[:, 1], "Cluster": clusters})
        else:
            fig = px.scatter(
                        x=embs_2d[:, 0], 
                        y=embs_2d[:, 1], 
                        color=clusters, 
                        title="Clusters en 2D",
                        color_discrete_sequence=px.colors.qualitative.Pastel,  
                        labels={"x": "Dim 1", "y": "Dim 2"},
                        hover_data={"ID": ids, "Cluster": clusters}  # Mostrar solo ID y Cluster
                    )

        # Ajustar tamaño de los puntos (equivalente a `s=40`)
        fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="gray")))
        fig.show()
        input("Presiona Enter para cerrar...")
        if write:
            file_pt = f'{base_out_pt}/grafico_clusters.html'
            fig.write_html(file_pt)
            print(f"Saving clusters plot to {file_pt}")

#################################################################################
#################################################################################   
def val_clusters(confs):
    lconfs      = confs.val_clusters
    data_pt     = f'{confs.local_pt.colab if u_detect_environment()[0] else confs.local_pt.local}/data'  
    db_pt       = confs.db_pt.colab if u_detect_environment()[0] else confs.db_pt.local

    #---------------------------------------------------------------------------
    # Load data
    file_names  = u_loadJson(f'{data_pt}/{lconfs.model_ver}/embs_file_names.json')
    labels      = u_loadJson(f'{data_pt}/{lconfs.model_ver}/labels.json')
    embs_2d     = torch.load(f'{data_pt}/{lconfs.model_ver}/embeddings_2d.pt', weights_only=False)
    embs        = torch.load(f'{data_pt}/{lconfs.model_ver}/embeddings.pt', weights_only=False) 
    history     = u_loadYaml(f'{data_pt}/{lconfs.model_ver}/clustering_history.yml')
    visualize_clusters(embs_2d, labels, f'{data_pt}/{lconfs.model_ver}', True, file_names)
    #---------------------------------------------------------------------------
    # Find mean image
    # find_mean_image(embs, labels, file_names, db_pt, history, lconfs.n_images,
    #                f'{data_pt}/{lconfs.model_ver}/imgs')

################################################################################
################################################################################
def find_mean_image(embs, labels, file_names, db_pt, history, n, save_path):
    names_df    = pd.DataFrame({'file_name': file_names, 'label': labels})
    embs_df     = pd.DataFrame(embs)
    metric      = 'euclidean'

    #---------------------------------------------------------------------------
    # Select clusters
    for cluster in tqdm(range(history.n_clusters)):
        tmp_embs    = embs_df[names_df.label == cluster] 
        tmp_files   = names_df.file_name[names_df.label == cluster] 
        avg         = tmp_embs.mean(axis=0).values

        distances           = cdist(tmp_embs.values, avg.reshape(1, -1), metric=metric).flatten()
        distances_sorted    = np.argsort(distances)

        nearests            = distances_sorted[:n]
        nearest_files       = tmp_files.iloc[nearests].values

        outliers            = distances_sorted[-n:]
        outlier_files       = tmp_files.iloc[outliers].values


        fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)  

        for i, file in enumerate(nearest_files):
            img = mpimg.imread(db_pt + file)
            axes[0, i].imshow(img)
            axes[0, i].axis("off")  

        for i, file in enumerate(outlier_files):
            img = mpimg.imread(db_pt + file)
            axes[1, i].imshow(img)
            axes[1, i].axis("off")  

        u_mkdir(save_path)
        plt.savefig(f'{save_path}/cluster_{cluster}_dst_{metric}.png', bbox_inches='tight', dpi=200)  
        
