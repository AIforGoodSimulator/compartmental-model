# install.packages("socialmixr")
library('socialmixr')
setwd("c:/Users/user/Documents/Python/AI4Good/compartmental-model-master/compartmental-model/Parameters")
source('ContactMatrices/contact_matrices.R')


data(polymod)
ageLims16 <- c(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75)
ageLims8 <- c(0, 10, 20, 30, 40, 50, 60, 70)

C16 <- as.data.frame(contact_matrix(polymod, countries = "United Kingdom", age.limits = ageLims16))
C8 <- as.data.frame(contact_matrix(polymod, countries = "United Kingdom", age.limits = ageLims8))
# C <- C/max(C) #scaling polymod - don't!


# 16 ages matrix
nCats16 <- length(ageLims16)
C16 <- C16[1:nCats16,1:nCats16]

# 8 ages matrix
nCats8 <- length(ageLims8)
C8 <- C8[1:nCats8,1:nCats8]


# nCats <- 4
# C_subset <- C[1:nCats,1:nCats]
# C_subset
# C <- as.matrix(C)
ageLimsGen <- c(0, 10, 20, 30, 40, 50, 60, 70, 80)
C8_generated <- generate_contact_matrix('Moria',ageLimsGen,C16)
C8_generated

C16[1:4,1:4]
C8[1:2,1:2]
C8_generated[1:2,1:2]

C8_generated - C8

#polymod is daily data
