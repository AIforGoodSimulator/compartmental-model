setwd("c:/Users/user/Documents/Python/AI4Good/compartmental-model-master/compartmental-model/Parameters")



generate_contact_matrix <- function(campName,age_limits,contact_matrix){
  
  camp_params <- read.csv("camp_params.csv")
  names(camp_params)[1] <- 'Variable'
  camp_params_pop <- camp_params[camp_params['Variable'] == 'Population_structure',]
  camp_params_pop_for_camp <- camp_params_pop[camp_params_pop['Camp']==campName,]$Value
  
  
  
  p <- numeric(16)
  
  
  
  
  # we have 1-16 in 5 year bands
  
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
  
return(M)
}


camp_name <- 'Moria'
ageLimits <- c(0,10,20,30,40,50,60,70,80)
wuhan_matrix <- read.csv('CM_wuhan.csv')


generated_contact_matrix <- generate_contact_matrix(camp_name,ageLimits,wuhan_matrix)

write.csv(generated_contact_matrix,  file = paste("Contact_matrix_",camp_name,".csv",sep=''))
