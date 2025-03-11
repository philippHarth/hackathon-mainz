from .pandas_stats_impl import Statistics 
import lib.FLY_preprocessing.fly_constants as FLY
import pandas as pd
import numpy as np


class FLYAggregateStatistics(Statistics):
    def __init__(self, index_data, compute_syncounts = True, reduced_syncount_set = False,  compute_clusters = True, compute_motifs = False, max_k = 19):
        self.enabled_syncounts = compute_syncounts
        self.reduced_syncount_set = reduced_syncount_set
        self.enabled_clusters = compute_clusters
        self.enabled_motifs = compute_motifs
        self.max_k = max_k

        super().__init__(index_data)


        self.columns = ["model_descriptor", "parameter_index",
                        
           "ALL_ALL", 
           "EXC_EXC", "EXC_INH", "EXC_UNKNOWN",  
           "INH_EXC", "INH_INH", "INH_UNKNOWN",
           "UNKNOWN_EXC", "UNKNOWN_INH", "UNKNOWN_UNKNOWN",
           
           "ALL_SOMA", "ALL_DEND", "ALL_AIS",
           "EXC_SOMA", "EXC_DEND", "EXC_AIS", 
           "INH_SOMA", "INH_DEND", "INH_AIS", 
           "UNKNOWN_SOMA", "UNKNOWN_DEND", "UNKNOWN_AIS",
           
           "ALL_EXC-SOMA", "ALL_EXC-DEND", "ALL_EXC-AIS", "ALL_INH-SOMA", "ALL_INH-DEND", "ALL_INH-AIS",
           "EXC_EXC-SOMA", "EXC_EXC-DEND", "EXC_EXC-AIS", "EXC_INH-SOMA", "EXC_INH-DEND", "EXC_INH-AIS",
           "INH_EXC-SOMA", "INH_EXC-DEND", "INH_EXC-AIS", "INH_INH-SOMA", "INH_INH-DEND", "INH_INH-AIS", 

           "EXC_INH-20-SOMA", "EXC_INH-20-DEND", "EXC_INH-20-AIS", "EXC_INH-20-UNKNOWN",
           "EXC_INH-21-SOMA", "EXC_INH-21-DEND", "EXC_INH-21-AIS", "EXC_INH-21-UNKNOWN",
           "EXC_INH-22-SOMA", "EXC_INH-22-DEND", "EXC_INH-22-AIS", "EXC_INH-22-UNKNOWN",
           "EXC_INH-23-SOMA", "EXC_INH-23-DEND", "EXC_INH-23-AIS", "EXC_INH-23-UNKNOWN",
           "EXC_INH-24-SOMA", "EXC_INH-24-DEND", "EXC_INH-24-AIS", "EXC_INH-24-UNKNOWN",
           "EXC_INH-25-SOMA", "EXC_INH-25-DEND", "EXC_INH-25-AIS", "EXC_INH-25-UNKNOWN",

           "INH-20_EXC-SOMA", "INH-20_EXC-DEND", "INH-20_EXC-AIS", "INH-20_EXC-UNKNOWN", "INH-20_INH-SOMA", "INH-20_INH-DEND", "INH-20_INH-AIS", "INH-20_INH-UNKNOWN",
           "INH-21_EXC-SOMA", "INH-21_EXC-DEND", "INH-21_EXC-AIS", "INH-21_EXC-UNKNOWN", "INH-21_INH-SOMA", "INH-21_INH-DEND", "INH-21_INH-AIS", "INH-21_INH-UNKNOWN",
           "INH-22_EXC-SOMA", "INH-22_EXC-DEND", "INH-22_EXC-AIS", "INH-22_EXC-UNKNOWN", "INH-22_INH-SOMA", "INH-22_INH-DEND", "INH-22_INH-AIS", "INH-22_INH-UNKNOWN",
           "INH-23_EXC-SOMA", "INH-23_EXC-DEND", "INH-23_EXC-AIS", "INH-23_EXC-UNKNOWN", "INH-23_INH-SOMA", "INH-23_INH-DEND", "INH-23_INH-AIS", "INH-23_INH-UNKNOWN",
           "INH-24_EXC-SOMA", "INH-24_EXC-DEND", "INH-24_EXC-AIS", "INH-24_EXC-UNKNOWN", "INH-24_INH-SOMA", "INH-24_INH-DEND", "INH-24_INH-AIS", "INH-24_INH-UNKNOWN",
           "INH-25_EXC-SOMA", "INH-25_EXC-DEND", "INH-25_EXC-AIS", "INH-25_EXC-UNKNOWN", "INH-25_INH-SOMA", "INH-25_INH-DEND", "INH-25_INH-AIS", "INH-25_INH-UNKNOWN",
           
           "CLUSTER-0", "CLUSTER-1", "CLUSTER-2", "CLUSTER-3", "CLUSTER-4", "CLUSTER-5", "CLUSTER-6", "CLUSTER-7", "CLUSTER-8", "CLUSTER-9", "CLUSTER-10",
           "CLUSTER-11", "CLUSTER-12", "CLUSTER-13", "CLUSTER-14", "CLUSTER-15", "CLUSTER-16", "CLUSTER-17", "CLUSTER-18", "CLUSTER-19", "CLUSTER-20",
           
           "MOTIF-1", "MOTIF-2", "MOTIF-3", "MOTIF-4", "MOTIF-5", "MOTIF-6", "MOTIF-7", "MOTIF-8",
           "MOTIF-9", "MOTIF-10", "MOTIF-11", "MOTIF-12", "MOTIF-13", "MOTIF-14", "MOTIF-15", "MOTIF-16"]

        self.dtypes = {
            "model_descriptor": "object",
            "parameter_index": "int64",
        }
        for col in self.columns[2:]:
            self.dtypes[col] = "float64"
        
        self.result = pd.DataFrame(columns=self.columns).astype(self.dtypes)

    

    def compute_synapse_counts(self, df, synapse_column):
        by_celltype, by_compartment, by_celltype_compartment = self.get_aggregate_synapse_counts(df, synapse_column)

        def count_by_ct(pre_ct, post_ct):
            pre_ct = list(pre_ct)
            condition = (by_celltype.index.get_level_values("pre_celltype").isin(pre_ct)) & (by_celltype.index.get_level_values("post_celltype").isin(post_ct))
            return by_celltype[condition][synapse_column].sum()
        
        def count_by_ct_compartment(pre_ct, post_ct, post_compartment):
            pre_ct = list(pre_ct)
            condition = (by_celltype_compartment.index.get_level_values("pre_celltype").isin(pre_ct)) & (by_celltype_compartment.index.get_level_values("post_celltype").isin(post_ct)) \
                & (by_celltype_compartment.index.get_level_values("post_compartment").isin(post_compartment))
            return by_celltype_compartment[condition][synapse_column].sum()
        
        def count_by_compartment(post_compartment):
            condition = by_celltype_compartment.index.get_level_values("post_compartment").isin(post_compartment)
            return by_celltype_compartment[condition][synapse_column].sum()
        
        def count_by_postct_compartment(post_celltype, post_compartment):
            condition = (by_celltype_compartment.index.get_level_values("post_celltype").isin(post_celltype)) & (by_celltype_compartment.index.get_level_values("post_compartment").isin(post_compartment))
            return by_celltype_compartment[condition][synapse_column].sum()
        
        def count_by_prect_compartment(pre_celltype, post_compartment):
            condition = (by_celltype_compartment.index.get_level_values("pre_celltype").isin(pre_celltype)) & (by_celltype_compartment.index.get_level_values("post_compartment").isin(post_compartment))
            return by_celltype_compartment[condition][synapse_column].sum()
        
        ## other way to get grouped counts with pandas
        ## multilevel_analysis.df_summary.groupby(["pre_celltype","post_compartment"]).agg({"empirical": "sum", "model-null": "sum"})

        if(self.reduced_syncount_set):
            
            self.result.loc[self.result.index[-1], "EXC_EXC"] = count_by_ct(FLY.EXC, FLY.EXC)
            self.result.loc[self.result.index[-1], "EXC_INH"] = count_by_ct(FLY.EXC, FLY.INH)
            self.result.loc[self.result.index[-1], "EXC_UNKNOWN"] = count_by_ct(FLY.EXC, FLY.UNKNOWN)
            self.result.loc[self.result.index[-1], "INH_EXC"] = count_by_ct(FLY.INH, FLY.EXC)
            self.result.loc[self.result.index[-1], "INH_INH"] = count_by_ct(FLY.INH, FLY.INH)
            self.result.loc[self.result.index[-1], "INH_UNKNOWN"] = count_by_ct(FLY.INH, FLY.UNKNOWN)
            self.result.loc[self.result.index[-1], "UNKNOWN_EXC"] = count_by_ct(FLY.UNKNOWN, FLY.EXC)
            self.result.loc[self.result.index[-1], "UNKNOWN_INH"] = count_by_ct(FLY.UNKNOWN, FLY.INH)
            self.result.loc[self.result.index[-1], "UNKNOWN_UNKNOWN"] = count_by_ct(FLY.UNKNOWN, FLY.UNKNOWN)

            self.result.loc[self.result.index[-1], "EXC_EXC-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.EXC, FLY.SOMA)
            self.result.loc[self.result.index[-1], "EXC_EXC-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.EXC, FLY.DEND)
            self.result.loc[self.result.index[-1], "EXC_EXC-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.EXC, FLY.AIS)
            self.result.loc[self.result.index[-1], "EXC_INH-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.INH, FLY.SOMA)
            self.result.loc[self.result.index[-1], "EXC_INH-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.INH, FLY.DEND)
            self.result.loc[self.result.index[-1], "EXC_INH-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.INH, FLY.AIS)
    
            self.result.loc[self.result.index[-1], "INH_EXC-SOMA"] = count_by_ct_compartment(FLY.INH, FLY.EXC, FLY.SOMA)
            self.result.loc[self.result.index[-1], "INH_EXC-DEND"] = count_by_ct_compartment(FLY.INH, FLY.EXC, FLY.DEND)
            self.result.loc[self.result.index[-1], "INH_EXC-AIS"] = count_by_ct_compartment(FLY.INH, FLY.EXC, FLY.AIS)
            self.result.loc[self.result.index[-1], "INH_INH-SOMA"] = count_by_ct_compartment(FLY.INH, FLY.INH, FLY.SOMA)
            self.result.loc[self.result.index[-1], "INH_INH-DEND"] = count_by_ct_compartment(FLY.INH, FLY.INH, FLY.DEND)
            self.result.loc[self.result.index[-1], "INH_INH-AIS"] = count_by_ct_compartment(FLY.INH, FLY.INH, FLY.AIS)

            return 
        
        self.result.loc[self.result.index[-1], "ALL_ALL"] = by_celltype[synapse_column].sum()
        self.result.loc[self.result.index[-1], "EXC_UNKNOWN"] = count_by_ct(FLY.EXC, FLY.UNKNOWN)
        self.result.loc[self.result.index[-1], "INH_UNKNOWN"] = count_by_ct(FLY.INH, FLY.UNKNOWN)
        self.result.loc[self.result.index[-1], "UNKNOWN_EXC"] = count_by_ct(FLY.UNKNOWN, FLY.EXC)
        self.result.loc[self.result.index[-1], "UNKNOWN_INH"] = count_by_ct(FLY.UNKNOWN, FLY.INH)
        self.result.loc[self.result.index[-1], "UNKNOWN_UNKNOWN"] = count_by_ct(FLY.UNKNOWN, FLY.UNKNOWN)

        #cond_other = by_celltype.index.get_level_values("pre_celltype").isin(FLY.OTHER) | by_celltype.index.get_level_values("post_celltype").isin(FLY.OTHER)
        #self.result.loc[self.result.index[-1], "OTHER"] = 0

        self.result.loc[self.result.index[-1], "EXC_EXC"] = count_by_ct(FLY.EXC, FLY.EXC)
        self.result.loc[self.result.index[-1], "EXC_INH"] = count_by_ct(FLY.EXC, FLY.INH)
        self.result.loc[self.result.index[-1], "INH_EXC"] = count_by_ct(FLY.INH, FLY.EXC)
        self.result.loc[self.result.index[-1], "INH_INH"] = count_by_ct(FLY.INH, FLY.INH)

        self.result.loc[self.result.index[-1], "ALL_SOMA"] = count_by_compartment(FLY.SOMA)
        self.result.loc[self.result.index[-1], "ALL_DEND"] = count_by_compartment(FLY.DEND)
        self.result.loc[self.result.index[-1], "ALL_AIS"] = count_by_compartment(FLY.AIS)

        self.result.loc[self.result.index[-1], "EXC_SOMA"] = count_by_prect_compartment(FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_DEND"] = count_by_prect_compartment(FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_AIS"] = count_by_prect_compartment(FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH_SOMA"] = count_by_prect_compartment(FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH_DEND"] = count_by_prect_compartment(FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH_AIS"] = count_by_prect_compartment(FLY.INH, FLY.AIS)
        self.result.loc[self.result.index[-1], "UNKNOWN_SOMA"] = count_by_prect_compartment(FLY.UNKNOWN, FLY.SOMA)
        self.result.loc[self.result.index[-1], "UNKNOWN_DEND"] = count_by_prect_compartment(FLY.UNKNOWN, FLY.DEND)
        self.result.loc[self.result.index[-1], "UNKNOWN_AIS"] = count_by_prect_compartment(FLY.UNKNOWN, FLY.AIS)

        self.result.loc[self.result.index[-1], "ALL_EXC-SOMA"] = count_by_postct_compartment(FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "ALL_EXC-DEND"] = count_by_postct_compartment(FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "ALL_EXC-AIS"] = count_by_postct_compartment(FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "ALL_INH-SOMA"] = count_by_postct_compartment(FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "ALL_INH-DEND"] = count_by_postct_compartment(FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "ALL_INH-AIS"] = count_by_postct_compartment(FLY.INH, FLY.AIS)

        self.result.loc[self.result.index[-1], "EXC_EXC-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_EXC-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_EXC-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "EXC_INH-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_INH-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_INH-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.INH, FLY.AIS)
  
        self.result.loc[self.result.index[-1], "INH_EXC-SOMA"] = count_by_ct_compartment(FLY.INH, FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH_EXC-DEND"] = count_by_ct_compartment(FLY.INH, FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH_EXC-AIS"] = count_by_ct_compartment(FLY.INH, FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH_INH-SOMA"] = count_by_ct_compartment(FLY.INH, FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH_INH-DEND"] = count_by_ct_compartment(FLY.INH, FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH_INH-AIS"] = count_by_ct_compartment(FLY.INH, FLY.INH, FLY.AIS)

        self.result.loc[self.result.index[-1], "EXC_INH-20-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.INH_20, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_INH-20-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.INH_20, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_INH-20-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.INH_20, FLY.AIS)
        self.result.loc[self.result.index[-1], "EXC_INH-20-UNKNOWN"] = count_by_ct_compartment(FLY.EXC, FLY.INH_20, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "EXC_INH-21-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.INH_21, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_INH-21-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.INH_21, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_INH-21-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.INH_21, FLY.AIS)
        self.result.loc[self.result.index[-1], "EXC_INH-21-UNKNOWN"] = count_by_ct_compartment(FLY.EXC, FLY.INH_21, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "EXC_INH-22-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.INH_22, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_INH-22-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.INH_22, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_INH-22-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.INH_22, FLY.AIS)
        self.result.loc[self.result.index[-1], "EXC_INH-22-UNKNOWN"] = count_by_ct_compartment(FLY.EXC, FLY.INH_22, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "EXC_INH-23-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.INH_23, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_INH-23-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.INH_23, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_INH-23-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.INH_23, FLY.AIS)
        self.result.loc[self.result.index[-1], "EXC_INH-23-UNKNOWN"] = count_by_ct_compartment(FLY.EXC, FLY.INH_23, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "EXC_INH-24-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.INH_24, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_INH-24-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.INH_24, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_INH-24-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.INH_24, FLY.AIS)
        self.result.loc[self.result.index[-1], "EXC_INH-24-UNKNOWN"] = count_by_ct_compartment(FLY.EXC, FLY.INH_24, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "EXC_INH-25-SOMA"] = count_by_ct_compartment(FLY.EXC, FLY.INH_25, FLY.SOMA)
        self.result.loc[self.result.index[-1], "EXC_INH-25-DEND"] = count_by_ct_compartment(FLY.EXC, FLY.INH_25, FLY.DEND)
        self.result.loc[self.result.index[-1], "EXC_INH-25-AIS"] = count_by_ct_compartment(FLY.EXC, FLY.INH_25, FLY.AIS)
        self.result.loc[self.result.index[-1], "EXC_INH-25-UNKNOWN"] = count_by_ct_compartment(FLY.EXC, FLY.INH_25, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "INH-20_EXC-SOMA"] = count_by_ct_compartment(FLY.INH_20, FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-20_EXC-DEND"] = count_by_ct_compartment(FLY.INH_20, FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-20_EXC-AIS"] = count_by_ct_compartment(FLY.INH_20, FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-20_EXC-UNKNOWN"] = count_by_ct_compartment(FLY.INH_20, FLY.EXC, FLY.UNKNOWN)
        self.result.loc[self.result.index[-1], "INH-20_INH-SOMA"] = count_by_ct_compartment(FLY.INH_20, FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-20_INH-DEND"] = count_by_ct_compartment(FLY.INH_20, FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-20_INH-AIS"] = count_by_ct_compartment(FLY.INH_20, FLY.INH, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-20_INH-UNKNOWN"] = count_by_ct_compartment(FLY.INH_20, FLY.INH, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "INH-21_EXC-SOMA"] = count_by_ct_compartment(FLY.INH_21, FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-21_EXC-DEND"] = count_by_ct_compartment(FLY.INH_21, FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-21_EXC-AIS"] = count_by_ct_compartment(FLY.INH_21, FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-21_EXC-UNKNOWN"] = count_by_ct_compartment(FLY.INH_21, FLY.EXC, FLY.UNKNOWN)
        self.result.loc[self.result.index[-1], "INH-21_INH-SOMA"] = count_by_ct_compartment(FLY.INH_21, FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-21_INH-DEND"] = count_by_ct_compartment(FLY.INH_21, FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-21_INH-AIS"] = count_by_ct_compartment(FLY.INH_21, FLY.INH, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-21_INH-UNKNOWN"] = count_by_ct_compartment(FLY.INH_21, FLY.INH, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "INH-22_EXC-SOMA"] = count_by_ct_compartment(FLY.INH_22, FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-22_EXC-DEND"] = count_by_ct_compartment(FLY.INH_22, FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-22_EXC-AIS"] = count_by_ct_compartment(FLY.INH_22, FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-22_EXC-UNKNOWN"] = count_by_ct_compartment(FLY.INH_22, FLY.EXC, FLY.UNKNOWN)
        self.result.loc[self.result.index[-1], "INH-22_INH-SOMA"] = count_by_ct_compartment(FLY.INH_22, FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-22_INH-DEND"] = count_by_ct_compartment(FLY.INH_22, FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-22_INH-AIS"] = count_by_ct_compartment(FLY.INH_22, FLY.INH, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-22_INH-UNKNOWN"] = count_by_ct_compartment(FLY.INH_22, FLY.INH, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "INH-23_EXC-SOMA"] = count_by_ct_compartment(FLY.INH_23, FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-23_EXC-DEND"] = count_by_ct_compartment(FLY.INH_23, FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-23_EXC-AIS"] = count_by_ct_compartment(FLY.INH_23, FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-23_EXC-UNKNOWN"] = count_by_ct_compartment(FLY.INH_23, FLY.EXC, FLY.UNKNOWN)
        self.result.loc[self.result.index[-1], "INH-23_INH-SOMA"] = count_by_ct_compartment(FLY.INH_23, FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-23_INH-DEND"] = count_by_ct_compartment(FLY.INH_23, FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-23_INH-AIS"] = count_by_ct_compartment(FLY.INH_23, FLY.INH, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-23_INH-UNKNOWN"] = count_by_ct_compartment(FLY.INH_23, FLY.INH, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "INH-24_EXC-SOMA"] = count_by_ct_compartment(FLY.INH_24, FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-24_EXC-DEND"] = count_by_ct_compartment(FLY.INH_24, FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-24_EXC-AIS"] = count_by_ct_compartment(FLY.INH_24, FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-24_EXC-UNKNOWN"] = count_by_ct_compartment(FLY.INH_24, FLY.EXC, FLY.UNKNOWN)
        self.result.loc[self.result.index[-1], "INH-24_INH-SOMA"] = count_by_ct_compartment(FLY.INH_24, FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-24_INH-DEND"] = count_by_ct_compartment(FLY.INH_24, FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-24_INH-AIS"] = count_by_ct_compartment(FLY.INH_24, FLY.INH, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-24_INH-UNKNOWN"] = count_by_ct_compartment(FLY.INH_24, FLY.INH, FLY.UNKNOWN)

        self.result.loc[self.result.index[-1], "INH-25_EXC-SOMA"] = count_by_ct_compartment(FLY.INH_25, FLY.EXC, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-25_EXC-DEND"] = count_by_ct_compartment(FLY.INH_25, FLY.EXC, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-25_EXC-AIS"] = count_by_ct_compartment(FLY.INH_25, FLY.EXC, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-25_EXC-UNKNOWN"] = count_by_ct_compartment(FLY.INH_25, FLY.EXC, FLY.UNKNOWN)
        self.result.loc[self.result.index[-1], "INH-25_INH-SOMA"] = count_by_ct_compartment(FLY.INH_25, FLY.INH, FLY.SOMA)
        self.result.loc[self.result.index[-1], "INH-25_INH-DEND"] = count_by_ct_compartment(FLY.INH_25, FLY.INH, FLY.DEND)
        self.result.loc[self.result.index[-1], "INH-25_INH-AIS"] = count_by_ct_compartment(FLY.INH_25, FLY.INH, FLY.AIS)
        self.result.loc[self.result.index[-1], "INH-25_INH-UNKNOWN"] = count_by_ct_compartment(FLY.INH_25, FLY.INH, FLY.UNKNOWN)


    def compute_motifs(self, syncount_matrix, discrete):
        deviations = self.motif_statistics_gpu.get_motif_distribution(syncount_matrix, discrete)
        #deviations = self.get_motif_distribution(syncount_matrix, discrete)
        for k in range(0, 16):
            self.result.loc[self.result.index[-1], f"MOTIF-{k+1}"] = deviations[k]