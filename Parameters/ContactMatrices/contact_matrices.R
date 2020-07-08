#setwd("c:/Users/user/Documents/Python/AI4Good/compartmental-model-master/compartmental-model/Parameters")



generate_contact_matrix <- function(campName,age_limits,contact_matrix){
  
  camp_params <- read.csv("camp_params_2.csv")
  names(camp_params)[1] <- 'Camp'
  
  camp_params_pop_for_camp <- camp_params[camp_params['Camp']==campName,]$Population_structure
  
  
  
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
      sump <- sum(p[ (ind_limits[rr]+1):ind_limits[rr+1] ]) # * sum(p[ (ind_limits[cc]+1):ind_limits[cc+1] ])
      # probability in this category
      for(jj in (ind_limits[cc]+1):ind_limits[cc+1]){
        for(ii in (ind_limits[rr]+1):ind_limits[rr+1]){
          V2 <- V2 + contact_matrix[ii,jj]*p[ii]/sump # *p[jj]
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


camp_name <- 'Haman-al-Alil'
ageLimits <- c(0,10,20,30,40,50,60,70,80)
inputContactMatrix <- read.csv('ContactMatrices/IraqCM.csv')


generated_contact_matrix <- generate_contact_matrix(camp_name,ageLimits,inputContactMatrix)

write.csv(generated_contact_matrix,  file = paste("Contact_matrix_",camp_name,".csv",sep=''))
