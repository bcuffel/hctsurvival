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

## Correlations

#Examine survival across the entire sample
s <- survfit(hct_survival ~ 1, data= d)
s

ggsurvfit(s, type = "survival") +
  add_risktable() +
  scale_x_continuous(expand=c(0, 1), limits = c(0, 36), breaks = seq(0, 36, 6))

## Explore predictors of censored survival data
f <- d %>% select(where(is.factor)) %>% names()
s <- f %>% map(function(x) {
#     x <- f[1]
     y <- paste0("hct_survival ~ ",x)
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
    y <- paste0("hct_survival ~ ",x)
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

##----------------------------------------------------------------------------------------------------------##
## Machine Learning with Mboost
##----------------------------------------------------------------------------------------------------------##
## Initial Splits ##
set.seed(123)
splits <- initial_validation_split(d)
validate <- validation_set(splits)
training_set <- training(splits)
survival_metrics <- metric_set(concordance_survival,brier_survival_integrated, brier_survival,
                               roc_auc_survival)
evaluation_time_points <- seq(0, 72, 24)

## Preprocessing
recipe_boost <- recipe(hct_survival ~ ., data=training_set) %>%
    update_role(ID,new_role = "ID") %>%
    step_impute_knn(all_numeric_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_zv(all_predictors())
recipe_cox <- recipe(hct_survival ~ ., data=training_set) %>%
    update_role(ID,new_role = "ID") %>%
    step_impute_knn(all_numeric_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_dummy(all_nominal_predictors()) %>%  
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors()) ## Normalize the dummy variables

recipe_umap <- recipe(hct_survival ~ ., data=training_set) %>%
    update_role(ID,new_role = "ID") %>%
    step_impute_knn(all_numeric_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_dummy(all_nominal_predictors(),-risk_group) %>%  
    step_zv(all_predictors()) %>%
    step_umap(all_numeric_predictors(), outcome = vars(risk_group), num_comp = tune(),
              neighbors=15) %>%
    step_normalize(all_numeric_predictors()) ## Normalize the dummy variables

#trained <- recipe_umap %>% prep() %>% bake(new_data = training(splits))
recipe_cox_prepca <- recipe(hct_survival ~ ., data=training_set) %>%
    update_role(ID,new_role = "ID") %>%
    step_impute_knn(all_numeric_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_dummy(all_nominal_predictors(),-race_group,-sex_match,-ethnicity) %>%  ## Hold 3 nominal predictors out
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors()) ## Normalize the dummy variables
recipe_cox_pca <- recipe_cox_prepca %>% step_pca(all_numeric_predictors(),num_comp = 12) %>% step_dummy(race_group,sex_match,ethnicity)

comment <- function() {
recipe_cox_pca_trained <- prep(recipe_cox_pca)
tidied <- tidy(recipe_cox_pca_trained,7)
1:12 %>% walk(function(x) {g <- tidied %>%
  filter(component %in% paste0("PC", x)) %>%
  group_by(component) %>%
  top_n(8, abs(value)) %>%
  ungroup() %>%
  mutate(terms = reorder_within(terms, abs(value), component)) %>%
  ggplot(aes(abs(value), terms, fill = value > 0)) +
  geom_col() +
  facet_wrap(~component, ncol=1,scales = "free_y") +
  scale_y_reordered() +
  labs(title = paste0("Principal Compenents ",x),
    x = "Absolute value of contribution",
    y = NULL, fill = "Positive?"
  )
print(g)
}
)
}
## Specify the candidate models
cox_spec <- boost_tree(mtry=5,min_n=tune(),trees=tune(),tree_depth = tune(),loss_reduction = .1) %>%
    set_engine('mboost') %>% set_mode('censored regression')
weibull_spec <- boost_tree(mtry=25,min_n=tune(),trees=tune(),tree_depth = tune(),loss_reduction = .11) %>%
    set_engine('mboost',family=Weibull()) %>%  set_mode('censored regression')
lognormal_spec <- boost_tree(mtry=25,min_n=tune(),trees=tune(),tree_depth = tune(),loss_reduction = .11) %>%
    set_engine('mboost',family=Lognormal()) %>%  set_mode('censored regression')
loglog_spec <- boost_tree(mtry=25,min_n=tune(),trees=tune(),tree_depth = tune(),loss_reduction = .1) %>% 
    set_engine('mboost',family=Loglog()) %>%  set_mode('censored regression')

cox_glmnet <- proportional_hazards(penalty=tune()) %>% set_engine('glmnet') %>% set_mode('censored regression')

## Set Up the Grid
grid <- workflow() %>%
    add_recipe(recipe_boost) %>%
    add_model(weibull_spec) %>% 
    extract_parameter_set_dials() %>% grid_space_filling(size=25)
grid_glmnet  <-  grid_regular(penalty(range= c(1e-3,.8),trans=NULL),levels=10)
## Tuning
grid_ctrl <-
   control_grid(
      save_pred = FALSE,
      parallel_over = "resamples",
      save_workflow = FALSE,
      verbose=FALSE
   )

## Workflow_set
survival_workflow_set <-
      workflow_set(
      preproc = list(boost = recipe_boost),
      models = list(cox = cox_spec,weibull = weibull_spec,lognormal=lognormal_spec,loglog=loglog_spec))
glmnet_workflow_set <-
      workflow_set(
      preproc = list(glmnet = recipe_cox),
      models = list(cox = cox_glmnet))
all_workflows <- bind_rows(survival_workflow_set,glmnet_workflow_set)

all_workflows <- 
  all_workflows %>% 
  option_add(grid = grid_glmnet, id = "glmnet_cox") %>%  
  option_add(grid = grid, id = "boost_weibull") %>% 
  option_add(grid = grid, id = "boost_lognormal") %>% 
  option_add(grid = grid, id = "boost_loglog")
plan(multisession,workers=3)
all_workflows <-
   all_workflows %>%
   workflow_map(fn = "tune_grid",
      resamples = validate,
      metrics = survival_metrics,
      eval_time = evaluation_time_points,
      control = grid_ctrl
   )
plan(sequential)
all_workflows %>%
   rank_results() %>%
   filter(.metric == "concordance_survival") %>%
   select(model, .config, concordance = mean, rank)
all_workflows %>%
   rank_results() %>%
   filter(.metric == "brier_integrated_survival") %>%
   select(model, .config, concordance = mean, rank)

autoplot(
   all_workflows,
   rank_metric = "concordance_survival",  # <- how to order models
   metric = "concordance_survival",       # <- which metric to visualize
   select_best = TRUE     # <- one point per workflow
) +
   geom_text(aes(y = mean - .1, label = wflow_id), angle = 90, hjust = 1) +
   lims(y = c(0, 1)) +
   theme(legend.position = "none")
cox_best <- 
   all_workflows %>% 
   extract_workflow_set_result("boost_cox") %>% select_best(metric = "concordance_survival")
cox_test_results <-
   all_workflows %>%
   extract_workflow("boost_cox") %>%
   finalize_workflow(cox_best) %>%
   last_fit(split = splits,eval_time = evaluation_time_points,metrics=survival_metrics)

collect_metrics(cox_test_results)
collect_predictions(cox_test_results)

## --------------------------------------------------------------------------------------------- ##
## Final Analyses Using Cox Model
## --------------------------------------------------------------------------------------------- ##
## GLMNET Cox Model without Principal Components
cox_wkflow <- workflow() %>%
    add_recipe(recipe_umap) %>%
    add_model(cox_spec)

grid_ctrl <-
   control_grid(
    save_pred = FALSE,
    parallel_over = "resamples",
    save_workflow = FALSE,
    verbose=FALSE
   )
st <- Sys.time();print(st)
plan(multisession, workers=3)
grid_cox <- cox_wkflow %>% extract_parameter_set_dials %>% grid_space_filling(25)
#grid_cox <- grid_regular(penalty(range = c(1e-4, .10),trans=NULL),levels= 5)
cox_results <- tune_grid(
    cox_wkflow,
    resamples = validate,
    grid = grid_cox,
    metrics = survival_metrics,
    eval_time = evaluation_time_points,
    control = grid_ctrl
)
et <- Sys.time()
et-st
plan(sequential)
show_best(cox_results, metric = "concordance_survival", n = 15)
show_best(cox_results, metric = "brier_survival_integrated",n=15)

param_best <- select_best(cox_results, metric = "concordance_survival")
last_cox_wflow <- finalize_workflow(cox_wkflow, param_best)
last_cox_fit <- last_fit(
    last_cox_wflow, 
    split = splits,
    metrics = survival_metrics,
    eval_time = evaluation_time_points)
collect_metrics(last_cox_fit) %>% filter(.metric == "concordance_survival")
last_cox_model <- extract_workflow(last_cox_fit)
pred_censored <-  predict(last_cox_model, new_data= testing(splits),type="linear_pred")
pred_submission <- predict(last_cox_model, new_data= d1,type="linear_pred")
p <- bind_cols(testing(splits),pred_censored,pred_risk = exp(pred_censored$.pred_linear_pred))
#glimpse(p)
summary(p$pred_risk)

## GLMNET Model with Principal Components
plan(multisession)
grid_ctrl <-
   control_grid(
      save_pred = FALSE,
      parallel_over = "resamples",
      save_workflow = FALSE,
      verbose=FALSE
   )
cox_wkflow <- workflow() %>%
    add_recipe(recipe_cox_pca) %>%
    add_model(cox_glmnet)
st <- Sys.time(); print(st)
grid_cox <- grid_regular(penalty(range = c(1e-4, .10),trans=NULL),levels= 5)
cox_results <- tune_grid(
  cox_wkflow,
  resamples = validate,
  grid = grid_cox,
  metrics = survival_metrics,
  eval_time = evaluation_time_points,
  control = grid_ctrl
)
et <- Sys.time()
et-st
show_best(cox_results, metric = "concordance_survival", n = 15)
show_best(cox_results, metric = "brier_survival_integrated",n=15)
param_best <- select_best(cox_results, metric = "concordance_survival")
last_cox_wflow <- finalize_workflow(cox_wkflow, param_best)
last_cox_fit <- last_fit(
  last_cox_wflow,
  split = splits,
  metrics = survival_metrics,
  eval_time = evaluation_time_points)
collect_metrics(last_cox_fit) %>% filter(.metric == "concordance_survival")
last_cox_model <- extract_workflow(last_cox_fit)
pred_censored <- predict(last_cox_model, new_data= testing(splits),type="linear_pred")
ppca <- bind_cols(testing(splits),pred_censored,pred_risk = exp(pred_censored$.pred_linear_pred))
#glimpse(p)
summary(ppca$pred_risk)

## GLMNET Model with Principal Components and Strata
plan(multisession)
grid_ctrl <-
   control_grid(
      save_pred = FALSE,
      parallel_over = "resamples",
      save_workflow = FALSE,
      verbose=FALSE
   )
cox_wkflow <- workflow() %>%
    add_recipe(recipe_cox_pca) %>%
    add_model(cox_glmnet,formula = hct_survival ~ . + strata(race_group)) %>%
    fit(training_set)

st <- Sys.time(); print(st)
grid_cox <- grid_regular(penalty(range = c(1e-4, .10),trans=NULL),levels= 5)
cox_results <- tune_grid(
  cox_wkflow,
  resamples = validate,
  grid = grid_cox,
  metrics = survival_metrics,
  eval_time = evaluation_time_points,
  control = grid_ctrl
)
et <- Sys.time()
et-st
show_best(cox_results, metric = "concordance_survival", n = 15)
show_best(cox_results, metric = "brier_survival_integrated",n=15)
param_best <- select_best(cox_results, metric = "concordance_survival")
last_cox_wflow <- finalize_workflow(cox_wkflow, param_best)
last_cox_fit <- last_fit(
  last_cox_wflow,
  split = splits,
  metrics = survival_metrics,
  eval_time = evaluation_time_points)
collect_metrics(last_cox_fit) %>% filter(.metric == "concordance_survival")
last_cox_model <- extract_workflow(last_cox_fit)
pred_censored <- predict(last_cox_model, new_data= testing(splits),type="linear_pred")
ppca <- bind_cols(testing(splits),pred_censored,pred_risk = exp(pred_censored$.pred_linear_pred))
#glimpse(p)
summary(ppca$pred_risk)




## Tune GLMNET Model with Principal Components and Interactions
plan(multisession)
grid_ctrl <-
    control_grid(
        save_pred = FALSE,
        parallel_over = "resamples",
        save_workflow = FALSE,
        verbose=FALSE)
recipe_cox_pcainteract <- recipe_cox_pca %>% step_interact(~ starts_with("race_group"):starts_with("PC")) %>%
    step_interact(~ starts_with("sex_match"):starts_with("PC")) %>%
    step_interact(~ starts_with("ethnicity"):starts_with("PC"))
cox_wkflow_interact <- workflow() %>%
    add_recipe(recipe_cox_pcainteract) %>%
    add_model(cox_spec)
st <- Sys.time(); print(st)
grid_cox <- grid_regular(penalty(range = c(1e-4, .10),trans=NULL),levels= 5)
cox_results_interact <- tune_grid(
    cox_wkflow_interact,
    resamples = validate,
    grid = grid_cox,
    metrics = survival_metrics,
    eval_time = evaluation_time_points,
    control = grid_ctrl)
et <- Sys.time()
et-st
show_best(cox_results_interact, metric = "concordance_survival", n = 15)
show_best(cox_results_interact, metric = "brier_survival_integrated",n=15)
param_best <- select_best(cox_results_interact, metric = "concordance_survival")
last_cox_wflow <- finalize_workflow(cox_wkflow_interact, param_best)
last_cox_fit <- last_fit(
    last_cox_wflow,
    split = splits,
    metrics = survival_metrics,
    eval_time = evaluation_time_points)
collect_metrics(last_cox_fit) %>% filter(.metric == "concordance_survival")
last_cox_model <- extract_workflow(last_cox_fit)
pred_censored <- predict(last_cox_model, new_data= testing(splits),type="linear_pred")
pinteract <- bind_cols(testing(splits),pred_censored,pred_risk = exp(pred_censored$.pred_linear_pred))
#glimpse(pinteract)
summary(pinteract$pred_risk)



## Tune Mboost Cox Model with principal components
plan(multisession)
grid_ctrl <-
   control_grid(
      save_pred = FALSE,
      parallel_over = "resamples",
      save_workflow = FALSE,
      verbose=FALSE
   )
cox_wkflow <- workflow() %>%
    add_recipe(recipe_cox_pca) %>%
    add_model(cox_spec)
st <- Sys.time(); print(st)
#grid_cox <- cox_wkflow %>% extract_parameter_set_dials %>% grid_space_filling(size=12)
grid <-    cox_wkflow %>% extract_parameter_set_dials %>% grid_space_filling(size=12)

#print(grid_cox)
cox_results <- tune_grid(
  cox_wkflow,
  resamples = validate,
  grid = grid,
  metrics = survival_metrics,
  eval_time = evaluation_time_points,
  control = grid_ctrl
)
et <- Sys.time()
et-st
show_best(cox_results, metric = "concordance_survival", n = 15)
show_best(cox_results, metric = "brier_survival_integrated",n=15)
param_best <- select_best(cox_results, metric = "concordance_survival")
last_cox_wflow <- finalize_workflow(cox_wkflow, param_best)
last_cox_fit <- last_fit(
  last_cox_wflow,
  split = splits,
  metrics = survival_metrics,
  eval_time = evaluation_time_points
)
collect_metrics(last_cox_fit) %>% filter(.metric == "concordance_survival")
last_cox_model <- extract_workflow(last_cox_fit)
pred_censored <- predict(last_cox_model, new_data= testing(splits),type="linear_pred")
p <- bind_cols(testing(splits),pred_censored,pred_risk = exp(pred_censored$.pred_linear_pred))
#glimpse(p)
summary(p$pred_risk)

## Plot Risk Scores

cbind(predict_cox = p$pred_risk,predict_boost = pboost$pred_risk,predict_cox_interact=pinteract$pred_risk) %>%
    ggplot(aes(x=predict_cox) +
           geom_point(aes(y=predict_boost),colour = mckinsey[1],alpha=.5)
           geom_point(aes(y=predict_boost),colour = mckinsey[2],alpha=.5)
