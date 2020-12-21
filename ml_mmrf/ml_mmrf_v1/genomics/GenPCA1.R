##create fpkm & design csv's for use in downstream analyses (like seurat)

#copies of data files needed for this script can be found: /afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/ia12/datafiles_reproduce_gene_expression_analysis/create_csv_fpkm_design_limited_baseline_cd138

#### load libraries, set wd ####
rm(list=ls())
library(tidyverse)
library(DESeq2)
library(pheatmap)
library(org.Hs.eg.db)
library(data.table)

setwd("./")

#### read in genetic data from MMRF ####

##fpkm
fpkm <- fread("/afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/ia15/MMRF_CoMMpass_IA15a_E74GTF_Cufflinks_Gene_FPKM.txt", stringsAsFactors = FALSE) #downloaded from CoMMpass
fpkm <- fpkm[,Location:=NULL] #remove gene location column


##design/QC
QC.filename <- "/afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/ia15/MMRF_CoMMpass_IA15_Seq_QC_Summary.csv"
design <- read.csv(QC.filename, stringsAsFactors = FALSE)


####  limit design to BM CD138+ baseline samples, which were sent for RNA-seq ####

#add sample ID column to design
design <- design %>% rowwise() %>% mutate(sampleID = paste(strsplit(QC.Link.SampleName, "_")[[1]][1:4], collapse = "_"),
                                          cellType = paste(strsplit(QC.Link.SampleName, "_")[[1]][4:5], collapse = "_"))
#are all colnames from counts in sample ID col I created?
stopifnot(all(setdiff(colnames(fpkm), "GENE_ID") %in% design$sampleID))

# limit to BM CD138+ baseline samples, which were sent for RNA-seq
print("limiting design table to BM Cd138 positive cells (as only those were sent for RNA-seq)...")
design <- filter(design, cellType == "BM_CD138pos")

#limit to baseline samples
print("limiting design table to baseline samples...")
design <- design %>% filter(Visits..Reason_For_Collection == "Baseline")

#samples have multiple entries in design -> limit to entries that were sent for RNA
print("limiting design table to those samples sent for RNA-seq...")
design <- design %>% filter(MMRF_Release_Status == "RNA-Yes")

#keep only subset of columns from design
design <- design %>% dplyr::select(Patients..KBase_Patient_ID, sampleID, cellType, Visits..Reason_For_Collection, Batch, QC_Percent_Mitochondrial)
colnames(design)[colnames(design) == "Patients..KBase_Patient_ID"] <- "public_id"
colnames(design)[colnames(design) == "Visits..Reason_For_Collection"] <- "visit"


#### add patient outcome, demographic, and clinical info to design ####

#create survival outcome buckets
outcomes <- read.csv("/afs/csail.mit.edu/group/clinicalml/datasets/multiple_myeloma/ia15/CoMMpass_IA15_FlatFiles/MMRF_CoMMpass_IA15_STAND_ALONE_SURVIVAL.csv", stringsAsFactors = FALSE)
outcomes <- outcomes %>% mutate("mortality_2yr" = ifelse((90 < deathdy & deathdy <= 730 & !is.na(deathdy)), "nonsurvivor", ifelse((lstalive > 730 | deathdy > 730), "survivor", NA))) %>% dplyr::select(public_id, deathdy, lstalive, mortality_2yr) #need to include !is.na(deathdy) or you get NA's wherever deathdy was NA
outcomes <- outcomes %>% mutate("mortality_1yr" = ifelse((90 < deathdy & deathdy <= 365 & !is.na(deathdy)), "nonsurvivor", ifelse((lstalive > 365 | deathdy > 365), "survivor", NA))) %>% dplyr::select(public_id, mortality_2yr, mortality_1yr, deathdy)


## add info from baseline tensor to design (values should already be normalized... but I'm not sure they are)
baseline_clinical <- read.csv(file = "data/baseline_clinical_tensor.csv", stringsAsFactors = F, row.names = 1) #this file was created in process_commpass/paper_results_code/MLHC-Paper-Results_IA12.ipynb

## rename clinical tensor columns to just feature name
## save other info in a dataframe

#function to parse current column names
parse_tensor_colnames <- function(colname){
  parsed_name = strsplit(colname, "\\.\\.\\.")[[1]]
  if (length(parsed_name) > 4){
    return(c(parsed_name[1:3], paste(parsed_name[4:length(parsed_name)], collapse = "_")))
  } else return(parsed_name)
}

#save feature info
feature_info <- data.frame(t(data.frame(sapply(colnames(baseline_clinical), parse_tensor_colnames))))
colnames(feature_info) <- c("category", "type", "name", "description")

#rename columns to basic names
colnames(baseline_clinical) <- make.unique(as.character(feature_info$name))

#choose columns to join to design
baseline_clinical <- baseline_clinical %>% mutate("public_id" = toupper(rownames(baseline_clinical)))
baseline_clinical_sel <- baseline_clinical %>% dplyr::select(public_id, starts_with("d_pt_"), starts_with("d_tri_cf"), starts_with("d_lab_"), starts_with("ss_"))

#read in 'time to comorbidity' info
time_to_morbid = read.csv("data/time_to_comp.csv", row.names = 1) #created in chronos, /afs/csail.mit.edu/u/r/rpeyser/MM/unsupervised_analysis/script1_create_array_for_unsupervised_learning.ipynb
rownames(time_to_morbid) <- toupper(rownames(time_to_morbid))
colnames(time_to_morbid) <- sapply(colnames(time_to_morbid), function(x) strsplit(x, "\\.\\.\\.")[[1]][1])
colnames(time_to_morbid) <- gsub("ss_", "timeto_", colnames(time_to_morbid))
time_to_morbid <- time_to_morbid %>% mutate("public_id" = rownames(time_to_morbid))


##merge them into design
#keep all patients in design, so it includes all patients with valid RNA-seq data, though only those with outcomes data will be useful
design <- left_join(design, outcomes, by = "public_id")
design <- left_join(design, baseline_clinical_sel, by = c("public_id"))
design <- left_join(design, time_to_morbid, by = c("public_id"))

##design may somehow has duplicate entries - remove
design <- unique(design)


#### limit fpkm to those samples we retained in design ####
#how many sample unique to each fpkm and design now? 
setdiff(colnames(fpkm), design$sampleID) #the samples we are throwing out
length(setdiff(design$sampleID, colnames(fpkm))) #should be empty
samples.to.keep <- intersect(colnames(fpkm), design$sampleID)

stopifnot(length(colnames(fpkm)) == length(unique(colnames(fpkm))))
fpkm <- fpkm %>% dplyr::select(GENE_ID, samples.to.keep)


#### sum rows of FPKM with the same gene_ID ####
if(length(fpkm$GENE_ID) != length(unique(fpkm$GENE_ID))){
  fpkm <- fpkm[, lapply(.SD, sum, na.rm=TRUE), by=GENE_ID ]
}

#### write design, counts, and fpkm to pickle dir for easy re-use ####
write.csv(design, "./design_limit_baseline_CD138.csv", quote = FALSE, row.names = FALSE)
write.csv(fpkm, "./fpkm_limit_baseline_CD138.csv", quote = FALSE, row.names = FALSE)