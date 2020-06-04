# Script for creating an easy-to-work-with dataframe that includes
# the allele name, peptide, protein sequence, IC50 value, and an empty
# column for the convolution.
# Dataframe Format:
# index, peptide (eiip), protein_allele (eiip), conv (length), meas

library(tidyverse)
library(dplyr)
library(stringr)

peptide_data <- as.data.frame(read.csv("MHC_peptide_ic50.csv"))

peptide_data <- peptide_data %>%
  filter(species == "human")

protein_data <- as.data.frame(read.csv("MHC_protein_seqs.csv"))

CNN_data <- data.frame("mhc" = peptide_data %>%
  arrange(mhc) %>%
  pull(mhc))

CNN_data$IC50 <- peptide_data %>%
  arrange(mhc) %>%
  pull(meas)

protein_data$Protein.Sequence <- stringr::str_remove_all(protein_data$Protein.Sequence, '\\*')
protein_data$Protein.Sequence <- stringr::str_remove_all(protein_data$Protein.Sequence, '\\.')

pepk <- 1
protk <- 1
total_count <- 0
CNN_data$peptide <- ""
CNN_data$protein <- ""

CNN_data$peptide <- peptide_data %>%
  pull(sequence)

while (total_count <= nrow(peptide_data)) {
  if (CNN_data$mhc[pepk] == protein_data$Allele[protk]) {
    CNN_data$protein[pepk] <- protein_data$Protein.Sequence[protk]
    pepk <- pepk + 1
  }  else if (CNN_data$mhc[pepk] == protein_data$Allele[protk + 1]) {
    CNN_data$protein[pepk] <- protein_data$Protein.Sequence[protk + 1]
    pepk <- pepk + 1
    protk <- protk + 1
  }  else {
    pepk <- pepk + 1
  }
  total_count <- total_count + 1
}

CNN_data <- CNN_data %>%
  arrange(mhc) %>%
  filter(protein != "")

#max_seq_length <- max(nchar(CNN_data$protein))
seq <- 1

while (seq < nrow(CNN_data)) {
  if (nchar(CNN_data$protein[seq]) != 366) {
    padding <- paste(rep('L', times = (366 - nchar(CNN_data$protein[seq]))), collapse = "")
    CNN_data$protein[seq] <- paste(CNN_data$protein[seq], padding, sep = "")
  }
  seq <- seq + 1
}

CNN_with_count <- CNN_data %>%
  group_by(protein) %>%
  arrange(protein) %>%
  mutate("protSeqLen" = nchar(protein))

CNN_data$peptide_EIIP <- ""
CNN_data$protein_EIIP <- ""
CNN_data$conv_length <- ""

write.csv(CNN_data, "non-convolved-CNN_data.csv", row.names = FALSE)

# CONVERTING IC50 to pIC50
data <- read.csv("CNN_with_EIIP.csv")
View(data)
data$IC50 <- (-1*log(data$IC50))
write.csv(data, "CNN_with_EIIP.csv")


