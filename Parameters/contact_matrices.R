load("C:/Users/user/Downloads/contacts_china.rdata")

setwd("c:/Users/user/Documents/Python/AI_for_good/AI-for-good/Parameters")

camp_params <- read.csv("camp_params.csv")
names(camp_params)[1] <- 'Variable'
camp_params_pop <- camp_params[camp_params['Variable'] == 'Population_structure',]
camp_params_pop_for_camp <- camp_params_pop[camp_params_pop['Camp']=='Camp_2',]$Value



p <- numeric(16)


contact_matrix <- contacts_china$all

# we have 1-16 in 5 year bands

age_limits <- c(0,20,40,80)
n_categories <- length((age_limits))-1
ind_limits <- age_limits/5

for(cc in 1:n_categories){
  for(i in (ind_limits[cc]+1):ind_limits[cc+1]){
    p[i] <- camp_params_pop_for_camp[cc]/(ind_limits[cc+1] -  ind_limits[cc]) # assumes even dist within category
  }
}



M <- matrix(0,n_categories,n_categories)
for(rr in 1:n_categories){
  for(cc in 1:n_categories){
    V2 <- 0
    sump <- sum(p[ (ind_limits[cc]+1):ind_limits[cc+1] ]) * sum(p[ (ind_limits[rr]+1):ind_limits[rr+1] ])
    # probability in this category
    for(i in (ind_limits[cc]+1):ind_limits[cc+1]){
      for(j in (ind_limits[rr]+1):ind_limits[rr+1]){
        V2 <- V2 + contact_matrix[j,i]*p[i]*p[j]/sump
        # add up all contributions by weighted average
      }
    }
    M[rr,cc] <- V2
  }
}

M <- as.data.frame(M)
for(i in 1:length(M)){
  names(M)[i] <- paste('Age ',age_limits[i],'-',age_limits[i+1],sep = '')
}



write.csv(contact_matrix,  file = "Contact_matrix_wuhan.csv")
write.csv(M,  file = "Contact_matrix.csv")