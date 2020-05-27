setwd("c:/Users/user/Documents/Python/AI4Good/compartmental-model-master/compartmental-model/Parameters")
library(tidyr)
library(dplyr)

params <- read.csv("parameters.csv")
names(params)[1] <- 'ParamName'


paramFrame <- select(params,ParamName,Value,Type) %>%
              filter(Type=='Model Parameter')

latentMean     <- paramFrame[paramFrame$ParamName=='latent period',]$Value
RecoveryMean   <- paramFrame[paramFrame$ParamName=='infectious period',]$Value
HospMean       <- paramFrame[paramFrame$ParamName=='hosp period',]$Value
DeathICUMean   <- paramFrame[paramFrame$ParamName=='death period with ICU',]$Value
DeathNoICUMean <- paramFrame[paramFrame$ParamName=='death period',]$Value


randomParams <- data.frame('R0'                = rnorm(1000, mean= 4,              sd=1),
                            'LatentPeriod'     = rnorm(1000, mean= latentMean,     sd=1),
                            'RemovalPeriod'    = rnorm(1000, mean= RecoveryMean,   sd=1),
                            'HospPeriod'       = rnorm(1000, mean= HospMean,       sd=1),
                            'DeathICUPeriod'   = rnorm(1000, mean= DeathICUMean,   sd=1),
                            'DeathNoICUPeriod' = rnorm(1000, mean= DeathNoICUMean, sd=1)
)

randomParams[randomParams<0] <- NA
randomParams <- na.omit(randomParams)



write.csv(randomParams,  file = "GeneratedParams.csv")
