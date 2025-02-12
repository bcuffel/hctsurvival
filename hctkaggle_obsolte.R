## ------------------------------------------------------------------------------------------------##
## Kaggle Hematopoietic Cell Transplant Surivival Competition                                      ##
## ------------------------------------------------------------------------------------------------##
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(tidymodels))
library(data.table)
library(skimr)
library(probably)
library(naniar)
library(doParallel)
library(finetune)
library(hrbrthemes)
library(censored)
library(survival,quietly=TRUE)
library(ggsurvfit)
library(stringi)

path<-file.path(getwd(),"Data","statisticallearning","SurvivalPredictions")
mckinsey <- c("#000CA4","#8BBAFF","#CACACA","#111425","#DBE7F1","#7089D1","#0A76DB","#FAFAFA","#ffd08b")
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

## ------- Competion Data  ---------#
d <-  as_tibble(fread(file.path(path,"train.csv")))
d <- d %>% mutate_if(is.character,~ifelse(.=="","Unknown",as.character(.)))
d <- d %>% mutate(cmv_status = case_when(cmv_status == "+/+" ~ "Pos-Pos",
                                         cmv_status == "+/-" ~ "Pos-Neg",
                                         cmv_status == "-/+" ~ "Neg-Pos",
                                         cmv_status == "-/-" ~ "Neg-Neg",
                                         cmv_status == "Unknown" ~ "Unknown"))
d <- d %>% mutate(hct_survival = Surv(efs_time, efs), .keep = "unused")
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
d %>% skim() %>% yank("factor") %>% select(skim_variable,n_missing,top_counts) %>% data.frame
d %>% skim() %>% yank("numeric") %>% select(skim_variable,n_missing,mean,sd,p0,p50,p100,hist) %>% data.frame

## Correlations

#Examine survival across the entire sample
s <- survfit(Surv(time = efs_time, event = efs) ~ 1, data= d)
s

ggsurvfit(s, type = "survival") +
  add_risktable() +
  scale_x_continuous(expand=c(0, 1), limits = c(0, 36), breaks = seq(0, 36, 6))

## Explore predictors of censored survival data
f <- d %>% select(where(is.factor)) %>% names()
s <- f %>% map(function(x) {
    y <- paste0("Surv(time = efs_time, event = efs) ~ ",x)
    s <- survfit2(as.formula(y),data=d)
    g <- ggsurvfit(s, type = "survival") +
         labs(title=paste0("KM of EFS on ",x)) +
         scale_x_continuous(expand=c(0, 1), limits = c(0, 20), breaks = seq(0, 20, 2))
    return(list(s=s,g=g))
    })

## Print the KMs for selected factors
g1 <- map(s[29],'g') #ethnicity
g2 <- map(s[28],'g') #sex_match
g3 <- map(s[18],'g') #race
suppressWarnings(print(g1))
suppressWarnings(print(g2))
suppressWarnings(print(g3))


## Extract the survfit tables and examine the distribution of restricted mean survival time by factor level
stables <- map(s,'s') %>% map_df(~as.data.frame(summary(.x)$table))
stables$var <- as.factor(stri_split_fixed(rownames(stables),"=",simplify=TRUE)[,1])
stables$factors <- as.factor(stri_split_fixed(rownames(stables),"=",simplify=TRUE)[,2])
rownames(stables) <- 1:dim(stables)[1]

g <- stables %>%
    ggplot(aes(x=rmean,y=fct_reorder(var,rmean,.fun='sd'),group=var,color=var)) +
    geom_point(show.legend = FALSE) +
       theme_ipsum(grid="Y",base_size=10) +
       labs(title="RMST Across All Qualitative Factors",x="Restricted Mean Survival Time",y="Factor")
suppressWarnings(print(g))

g <- stables %>%
    ggplot(aes(x=median,y=fct_reorder(var,median,.fun='sd'),group=var,color=var)) +
    geom_point(show.legend = FALSE) +
       theme_ipsum(grid="Y",base_size=10) +
       labs(title="Median Survival Across All Qualitative Factors",x="Median Survival Time",y="Factor")
suppressWarnings(print(g))

## for numeric predictors
f <- d %>% select(where(is.numeric),-ID) %>% names()
s2 <- f %>% map(function(x) {
    y <- paste0("Surv(time = efs_time, event = efs) ~ ",x)
    s <- coxph(as.formula(y),data=d)
    })
## Extract the survfit tables and examine the hazard ratios for each predictor
#summary(s2[[1]])$coefficients
stables <- s2  %>% map_df(~as.data.frame(summary(.x)$coefficients))
stables$factors <- as.factor(rownames(stables))
rownames(stables) <- 1:dim(stables)[1]
g <- stables %>%
    ggplot() +
    geom_col(aes(x=z,y=fct_reorder(factors,z)),fill=mckinsey[1]) +
       theme_ipsum(grid="Y",base_size=10) +
       labs(title="Z scores for Hazard Ratios Across All Numeric Predictors of Survival",x="Z score",y="Predictor of Survival")
suppressWarnings(print(g))

f <- d %>% select(where(is.numeric),-ID) %>% names()
int_numeric_pvalue  <- f %>% map_dbl(function(x) {
## Screen for Potential Numeric Interactions
    x <- f[2]
    dtemp <- eval(parse(text = paste0("d[!is.na(d$", x, "),]")))
    y  <- paste0("Surv(time = efs_time, event = efs) ~ race_group")
    y2 <- paste0("Surv(time = efs_time, event = efs) ~ race_group + race_group:",x)
    s2 <- coxph(as.formula(y),data=dtemp)
    s3 <- coxph(as.formula(y2),data=dtemp)
    p <- anova(s2,s3)[[4]][[2]]
    return(p)
    })
int_numeric_pvalue <- data.frame(Interaction=f, pvalue = int_numeric_pvalue) %>% arrange(pvalue)

f <- d %>% select(where(is.factor),-ID) %>% names() 
int_factor_pvalue  <- f %>% map_dbl(function(x) {
## Screen for Potential Factor Interactions
    y  <- paste0("Surv(time = efs_time, event = efs) ~ race_group")
    y2 <- paste0("Surv(time = efs_time, event = efs) ~ race_group + race_group:",x)
    s2 <- coxph(as.formula(y),data=d)
    s3 <- coxph(as.formula(y2),data=d)
    p <- anova(s2,s3)[[4]][[2]]
    return(p)
    })
int_factor_pvalue <- data.frame(Interaction=f, pvalue = int_factor_pvalue) %>% arrange(pvalue)
top_list<- c(int_factor_pvalue$Interaction[1:3],int_numeric_pvalue$Interaction[1:3])

##----------------------------------------------------------------------------------------------------------##
## Machine Learning with Mboost
##----------------------------------------------------------------------------------------------------------##
## Initial Splits ##
set.seed(123)
splits <- initial_validation_split(d)
validate <- validation_set(splits)
survival_metrics <- metric_set(brier_survival_integrated, brier_survival,
                               roc_auc_survival, concordance_survival)
evaluation_time_points <- seq(0, 72, 24)

## Preprocessing (without interaction terms)
recipe_boost <- recipe(hct_survival ~ ., data=training(splits)) %>%
    update_role(ID,new_role = "ID") %>%
    step_dummy(all_nominal_predictors(),one_hot = TRUE) %>%
    step_zv(all_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02)

recipe_cox <- recipe(hct_survival ~ ., data=training(splits)) %>%
    update_role(ID,new_role = "ID") %>% 
    step_impute_knn(all_numeric_predictors()) %>%
    step_unknown(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())

## Specify the candidate models:  Cox, Weibull, Lognormal, and Loglog
## Giving up on glmnet following lots of effort.
#cox_spec <-        proportional_hazards(penalty=tune()) %>% set_engine('survival') %>%  set_mode('censored regression')
cox_spec <-       boost_tree(mtry=tune(),min_n=tune(),trees=tune())  %>% set_engine('mboost') %>% set_mode('censored regression')
Weibull_spec <-   boost_tree(mtry=tune(),min_n=tune(),trees=tune()) %>% set_engine('mboost',family=Weibull()) %>%  set_mode('censored regression')
Lognormal_spec <- boost_tree(mtry=tune(),min_n=tune(),trees=tune()) %>% set_engine('mboost',family=Lognormal()) %>%  set_mode('censored regression')
Loglog_spec <-    boost_tree(mtry=tune(),min_n=tune(),trees=tune()) %>% set_engine('mboost',family=Loglog()) %>%  set_mode('censored regression')

## Create a workflow set
aft <-
      workflow_set(
      preproc = list(aft = recipe_boost),
      models = list(cox = cox_spec,weibull = Weibull_spec,lognormal=Lognormal_spec,loglog=Loglog_spec))
print(aft)

## Tuning
num_cores <- parallel::detectCores(logical=FALSE) - 1
doParallel::registerDoParallel(cores = num_cores)
grid_ctrl <-
   control_grid(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = FALSE
   )
grid_results <-
   all_workflows %>%
   workflow_map(
      resamples = validate,
      grid = 10,
      metrics = survival_metrics,
      eval_time = evaluation_time_points,
      control = grid_ctrl,
      verbose=TRUE
   )
grid_results %>%
   rank_results() %>%
   filter(.metric == "concordance_survival") %>%
   select(wflow_id,model, .config, concordance = mean, rank) %>% print(n = Inf)

autoplot(
   grid_results,
   rank_metric = "concordance_survival",  # <- how to order models
   metric = "concordance_survival",       # <- which metric to visualize
   select_best = TRUE     # <- one point per workflow
) +
   geom_text(aes(y = mean-.01, label = wflow_id), angle = 90, hjust = 1) +
   lims(y = c(0, 1)) +
   theme(legend.position = "none")
## Fitting the Final Model - Performance against Test Data
## Note that this should come after the final tuning of the Cox Model Approach
best_results <-
   grid_results %>%
   extract_workflow_set_result("cox_model") %>%
   select_best(metric = "concordance_survival")

cox_test_results <-
   grid_results %>%
   extract_workflow("cox_model") %>%
   finalize_workflow(best_results) %>%
   last_fit(split = splits,metrics=survival_metrics,eval_time = evaluation_time_points)

collect_metrics(cox_test_results) # Result of concordance look good ~ .66 or .67

## Final Analyses Using Cox PH approach
recipe_cox_strata <- recipe(efs_time + efs  ~ .,data=training(splits)) %>%
    update_role(ID,new_role = "ID") %>%
    update_role(efs_time,efs, new_role = "outcome") %>% 
    step_impute_knn(all_numeric_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())
recipe_cox <- recipe(efs_time + efs ~ .,data=training(splits)) %>%
    update_role(ID,new_role = "ID") %>% 
    update_role(efs_time,efs, new_role = "outcome") %>% 
    step_impute_knn(all_numeric_predictors(),-efs_time) %>%
    step_unknown(all_nominal_predictors()) %>%
    step_novel(all_nominal_predictors()) %>% 
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors(),-efs_time) %>%
    step_interact(~ starts_with("race_group"):karnofsky_score +
                      starts_with("race_group"):age_at_hct +
                      starts_with("race_group"):comorbidity_score +
                      starts_with("race_group"):starts_with("prim_disease_hct") +
                      starts_with("race_group"):starts_with("conditioning_intensity") +
                      starts_with("race_group"):starts_with("dri_score"))

## Final Work Flow Cox Model with Stratification by Race Group
cox_wkflow_strata <- workflow() %>%
    add_recipe(recipe_boost) %>%
    add_model(cox_spec)

cox_rs_strata <- tune_grid(
  cox_wkflow_strata,
  resamples = validate,
  grid = 5,
  metrics = survival_metrics,
  eval_time = evaluation_time_points,
  control = grid_ctrl
)

db <- bake(prep(recipe_boost), new_data = training(splits))
f <- fit(cox_wkflow_strata, data = training(splits))
print(f)
dbtest <- bake(prep(recipe_cox_strata), new_data = testing(splits))
dbtest <- dbtest %>%  mutate(year_hct = as.integer(year_hct))
dbtest <- dbtest %>%  mutate(across(where(is.factor), ~ factor(.x, levels = levels(training(splits)[[cur_column()]]))))
str(dbtest)
best_lambda <- .1
preds <- predict(f, new_data = dbtest, type = "survival", eval_time = c(12, 24))
## Final Work Flow Cox Model Interactions by Race Group
f  <- formula(Surv(efs_time, efs) ~ .)
cox_wkflow <- workflow() %>%
    add_recipe(recipe_cox) %>%
    add_model(cox_spec,formula = f)
cox_rs <- tune_grid(
  cox_wkflow,
  resamples = validate,
  grid = 5,
  metrics = survival_metrics,
  eval_time = evaluation_time_points,
  control = grid_ctrl
)

show_best(cox_rs_strata, metric = "brier_survival_integrated", n = 5)
show_best(cox_rs_strata, metric = "concordance_survival", n = 15)
show_best(cox_rs_strata, metric = "roc_auc_survival", n = 15,eval_time = 24)
show_best(cox_rs_strata, metric = "roc_auc_survival", n = 15,eval_time = 48)
show_best(cox_rs_strata, metric = "roc_auc_survival", n = 15,eval_time = 72)

show_best(cox_rs, metric = "brier_survival_integrated", n = 5)
show_best(cox_rs, metric = "concordance_survival", n = 15)
show_best(cox_rs, metric = "roc_auc_survival", n = 15,eval_time = 24)
show_best(cox_rs, metric = "roc_auc_survival", n = 15,eval_time = 48)
show_best(cox_rs, metric = "roc_auc_survival", n = 15,eval_time = 72)


metrics <- collect_metrics(cox_rs_strata)
metrics %>% filter(.metric == "concordance_survival")

## Last Fit 
param_best <- select_best(cox_rs_strata, metric = "concordance_survival")
last_cox_wflow <- finalize_workflow(cox_wkflow_strata, param_best)
set.seed(1234)
last_cox_fit <- last_fit(
  last_cox_wflow, 
  split = splits,
  metrics = survival_metrics,
  eval_time = evaluation_time_points, 
)

collect_metrics(last_cox_fit) %>% 
  filter(.metric == "concordance_survival")

extract_workflow(last_cox_fit) %>% formula()
extract_recipe(last_cox_fit)
extract_fit_parsnip(last_cox_fit)


unregister_dopar()
doParallel::stopImplicitCluster()





