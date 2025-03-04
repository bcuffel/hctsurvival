## ------------------------------------------------------------------------------------------------##
## Kaggle Hematopoietic Cell Transplant Surivival Competition                                      ##
## ------------------------------------------------------------------------------------------------##
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(tidymodels))
library(data.table)
library(skimr)
library(probably)
library(naniar)
library(finetune)
library(hrbrthemes)
library(censored)
library(survival,quietly=TRUE)
library(ggsurvfit)
library(stringi)
library(future)
library(furrr)
library(textrecipes)
library(janitor)
library(tidytext)
library(embed)
path<-file.path(getwd(),"Data","statisticallearning","SurvivalPredictions")
mckinsey <- c("#000CA4","#8BBAFF","#CACACA","#111425","#DBE7F1","#7089D1","#0A76DB","#FAFAFA","#ffd08b")

## ------- Competion Data  ---------#
d <-  as_tibble(fread(file.path(path,"train.csv"))) ##[sample(.N,1000)])
d <- d %>% mutate_if(is.character,~ifelse(.=="","Unknown",as.character(.)))
d <- d %>% mutate(cmv_status = case_when(cmv_status == "+/+" ~ "Pos-Pos",
                                         cmv_status == "+/-" ~ "Pos-Neg",
                                         cmv_status == "-/+" ~ "Neg-Pos",
                                         cmv_status == "-/-" ~ "Neg-Neg",
                                         cmv_status == "Unknown" ~ "Unknown"))
bp <- quantile(d$efs_time,probs=seq(0,1,0.25))
d$risk_group <- cut(d$efs_time, breaks = bp ,include.lowest=TRUE,right=FALSE)
levels(d$risk_group) <- c("Lowest", "Low","High","Highest")

d <- d %>% mutate(hct_survival = Surv(efs_time, efs==1), .keep = "unused")
d <- d %>% mutate_if(is.character, as.factor)


## -- Test Data from Kaggle
d1 <- as_tibble(fread(file.path(path,"test.csv")))
d1 <- d1 %>% mutate_if(is.character,~ifelse(.=="","Unknown",as.character(.)))
d1 <- d1 %>% mutate(cmv_status = case_when(cmv_status == "+/+" ~ "Pos-Pos",
                                         cmv_status == "+/-" ~ "Pos-Neg",
                                         cmv_status == "-/+" ~ "Neg-Pos",
                                         cmv_status == "-/-" ~ "Neg-Neg",
                                         cmv_status == "Unknown" ~ "Unknown"))
d1 <- d1 %>% mutate_if(is.character, as.factor)

## ------------------------------------------------------------------------------------------------##
## Address Missing Values
## ------------------------------------------------------------------------------------------------##
miss_var_summary(d) %>% filter(n_miss > 0) %>% print(n=Inf)
ddd <- d %>% select(where(is.factor)) %>% summary()

## EDA
d %>% select(-hct_survival) %>% skim() %>% yank("factor") %>% select(skim_variable,n_missing,top_counts) %>% data.frame
d %>% select(-hct_survival) %>% skim() %>% yank("numeric") %>% select(skim_variable,n_missing,mean,sd,p0,p50,p100,hist) %>% data.frame

## ------------------------------------------------------------------------------------------------##
## Exploratory Data Analysis
## ------------------------------------------------------------------------------------------------##

#Examine survival across the entire sample
s <- survfit(hct_survival ~ 1, data= d)
s
##-------------------------------------------------------------------------------------------------##
## Split Data 
##-------------------------------------------------------------------------------------------------##
set.seed(123)
splits <- initial_validation_split(d,strata = risk_group)
validate <- validation_set(splits)
training_set <- training(splits)
survival_metrics <- metric_set(concordance_survival,brier_survival_integrated, brier_survival,
                               roc_auc_survival)
evaluation_time_points <- seq(0, 72, 24)
cv = vfold_cv(v = 5,data=training_set)

##-----------------------------------------------------------------------------------------------##
## Preprocessing 
##-----------------------------------------------------------------------------------------------##
recipe_umap <- recipe(hct_survival ~ ., data=training_set) %>%
    step_impute_knn(all_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_dummy(all_nominal_predictors(),-risk_group) %>%  
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors()) %>% ## Normalize the dummy variables
    step_umap(all_numeric_predictors(), outcome = vars(risk_group), num_comp = 5)

## --------------------------------------------------------------------------------------------- ##
## Model Specification 
## --------------------------------------------------------------------------------------------- ##
cox_spec <- boost_tree(mtry=5,min_n=40,trees=2000,tree_depth = 8,loss_reduction = .1) %>%
    set_engine('mboost') %>% set_mode('censored regression')
## --------------------------------------------------------------------------------------------- ##
## Workflow for the Model with UMAP dimensionality Reduction and Resampling
## --------------------------------------------------------------------------------------------- ##
cox_wkflow <- workflow() %>%
    add_recipe(recipe_umap) %>%
    add_model(cox_spec)
## --------------------------------------------------------------------------------------------- ##
## Resampling / Cross Validation 
## --------------------------------------------------------------------------------------------- ##
## Mboost Model with UMAP
plan(multisession)
cox_wkflow <- workflow() %>%
    add_recipe(recipe_umap) %>%
    add_model(cox_spec)
st <- Sys.time(); print(st)
cox_results <- cox_wkflow %>%
  fit_resamples(
    resamples = validate,
    metrics = metric_set(concordance_survival),
    control = control_resamples(save_pred = TRUE)
  )
et <- Sys.time()
et-st
collect_metrics(cox_results)

## --------------------------------------------------------------------------------------------- ##
## Fit a final workflow that allows for prediction (removing risk_group) 
## --------------------------------------------------------------------------------------------- ##
last_surv_fit <- fit(cox_wkflow, data = training_set)
last_surv_fit %>%
    extract_mold() %>%
    pluck("predictors") %>%
    colnames()
# Extract trained recipe
final_recipe <- last_surv_fit %>% extract_preprocessor() 
final_recipe <- final_recipe %>%
  update_role(risk_group, new_role = "id") %>%  # Mark as non-predictor
  update_role_requirements(role = "id", bake = FALSE)  # Ensure it is ignored at prediction time

# Create a new workflow with the updated recipe
final_workflow <- workflow() %>%
  add_recipe(final_recipe) %>%
  add_model(cox_spec)  # Use the same boosting model

# Fit the updated workflow to training data
last_surv_fit <- fit(final_workflow, data = training_set)

## --------------------------------------------------------------------------------------------- ##
# Apply the Trained Model to New Data
## --------------------------------------------------------------------------------------------- ##
pred_submission <- predict(last_surv_fit, d1, type = "linear_pred")
head(pred_submission)
write.csv(pred_submission,"submission.csv",row.names=FALSE)
