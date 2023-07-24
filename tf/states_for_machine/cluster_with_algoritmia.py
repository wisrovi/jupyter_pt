from states_for_machine.cluster_algoritmia import ClusterAlgoritmia
from state_machine.State import State
import pickle



class Cluster_with_algortimia(State):
    
    def execute(self, **kwargs: dict) -> dict:
        """
        method to create unsupervised clusters based on graph-based algorithms and direct hausdorff distance, which is based on spatial distance

        @type kwargs: dict
        @param kwargs: dict with data to process in specific the dataset

        @rtype: dict
        @returns: dict with dataset with clusters
        """  
        
        if not kwargs.get("dataset"):
            raise Exception("No calibrations in kwargs")

        all_lines_saved = kwargs.get("dataset").get("all_lines")
        all_names_saved = kwargs.get("dataset").get("all_names")

        if not kwargs.get("sensitive"):
            raise Exception("No sensitive in kwargs")

        sensitive = kwargs.get("sensitive")

        # create the clusters with algoritmia
        k_william = ClusterAlgoritmia(sensibility=sensitive)
        k_william.train(all_lines_saved)
        cluster = k_william.predict([0])

        # in the created clusters, those that have not been grouped must be added
        sumary_input = {}
        for clust, ids_cluster in enumerate(k_william.get_clusters):
            sumary_input[clust] = len(ids_cluster)

        others = []
    
        all_clusters = []
        all_ids = []
        all_groups = []
        for clust, ids_cluster in enumerate(k_william.get_clusters):
            
            group = []
            for id_line1, id_p in enumerate(ids_cluster):

                # convert each line to a list of points (x, y)
                linea1 = all_lines_saved[id_p]

                all_clusters.append(( linea1, clust))  
                all_ids.append( id_p )
                group.append(id_p)

            all_groups.append(group)
                            
        # looging for the lines that are not in the clusters and add to the new id cluster
        for id_other in set([i for i in range(len(all_names_saved))]) - set(all_ids):
            others.append( id_other )
                    
        others = sorted(list(set(others)))
        
        # convert each line to a list of points (x, y) of new cluster
        new_cluster = []
        new_id_cluster = len(k_william.get_clusters)
        for id_other in others:

            all_clusters.append((all_lines_saved[id_other] , new_id_cluster ))  
            
            all_ids.append( id_other )
            new_cluster.append( id_other )
        
        # now we split the cluster in 2 groups, X and y
        X = [ d[0] for d in all_clusters]
        y = [ d[1] for d in all_clusters]

        tmp = all_groups  # k_william.get_clusters
        if len(new_cluster)>0:
            tmp.append(new_cluster)

        kwargs["dataset"]["X"] = X
        kwargs["dataset"]["y"] = y
        kwargs["dataset"]["clusters"] = tmp # clusters

        return kwargs