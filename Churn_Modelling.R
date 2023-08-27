# Import libraries & dataset ----
library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)

#1. Import dataset.
raw <- read_delim('Churn_Modelling.csv') %>% view()
raw %>% glimpse()
df <- raw

#2. Remove unneeded columns.
df <- select(raw, -RowNumber, -CustomerId, -Surname)

df %>% view()
# ----------------------------- Data Preprocessing -----------------------------
df$Exited %>% table() %>% prop.table()
df %>% inspect_na()

distinct(df, Geography) %>% view()

df <- df %>%
  mutate(Geography = recode(Geography,
                            "Germany" = "1",
                            "France" = "2",
                            "Spain" = "3"
  ))

df %>% view()

df$Geography <- as.numeric(df$Geography)

class(df$Geography)

# IV (information values) 
iv <- df %>% 
  iv(y = 'Exited') %>% as_tibble() %>%
  mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))

# Exclude not important variables 
ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]] 

df.iv <- df %>% select(Exited,ivars)

df.iv %>% dim()

# woe binning 
bins <- df.iv %>% woebin("Exited")

bins$Geography %>% as_tibble() %>% view()
bins$Geography%>% woebin_plot()


# breaking data into train and test & converting into woe values
dt_list <- df.iv %>% 
  split_df("Exited", ratio = 0.8, seed = 123)

train_woe <- dt_list$train %>% woebin_ply(bins) 
test_woe <- dt_list$test %>% woebin_ply(bins)

names <- train_woe %>% names() %>% gsub("_woe","",.)                   
names(train_woe) <- names              ; names(test_woe) <- names
train_woe %>% inspect_na() %>% tail(2) ; test_woe %>% inspect_na() %>% tail(2)


# Multicollinearity ----

# coef_na
target <- "Exited"
features <- train_woe %>% select(-Exited) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")
glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

# VIF (Variance Inflation Factor) 
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = train_woe, family = "binomial")
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 

#####--------3. Build Churn model.

h2o.init()

train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
test_h2o <- test_woe %>% select(target,features) %>% as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}
model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)

h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'blue') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)

###------4. Compare model results for training and test sets.

test_woe$Exited %>% table()
train_woe$Exited %>% table()

###-----5. Evaluate and explain model results using ROC & AUC curves.

# Prediction & Confision Matrice
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = dt_list$test$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")

model %>% 
  h2o.confusionMatrix(test_h2o) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))



# Check overfitting ----
model %>% 
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)

model %>% h2o.performance(newdata = test_h2o) %>% h2o.precision() %>% view()

model %>% h2o.performance(newdata = test_h2o) %>% h2o.recall() %>% view()

model %>% h2o.performance(newdata = test_h2o) %>% h2o.F2() %>% view()

model %>% h2o.performance(newdata = test_h2o) %>% h2o.F0point5() %>% view()
