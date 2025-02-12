library(tidymodels)
library(censored)
library(survival)
library(modeldatatoo)
library(future)
library(furrr)
tidymodels_prefer()

d <- ovarian
d <- d %>% mutate(ovarian_survival = Surv(futime, fustat == 1), 
    .keep = "unused")

##----------------------------------------------------------------------------------------------------------##
## Machine Learning with Mboost
##----------------------------------------------------------------------------------------------------------##
## Initial Splits ##
set.seed(123)
splits <- initial_validation_split(d)
validate <- validation_set(splits)
train_set <- training(splits)
survival_metrics <- metric_set(brier_survival_integrated, brier_survival,
                               roc_auc_survival, concordance_survival)
evaluation_time_points <- seq(0, 1000,100)

## Model specifications
spec <- survival_reg(dist = tune()) %>% set_engine("survival") %>% set_mode("censored regression")
cox_glmnet <-   proportional_hazards(penalty=tune()) %>% set_engine('glmnet') %>% set_mode('censored regression')
cox_spec <-     boost_tree(mtry=tune(),min_n=tune(),trees=tune(),tree_depth=tune(),loss_reduction=tune())  %>% 
                set_engine('mboost') %>% set_mode('censored regression')
weibull_spec <- boost_tree(mtry=tune(),min_n=tune(),trees=tune(),tree_depth=tune(),loss_reduction=tune()) %>% 
                set_engine('mboost',family=Weibull()) %>%
                set_mode('censored regression')
lognormal_spec <- boost_tree(mtry=tune(),min_n=tune(),trees=tune(),tree_depth=tune(),loss_reduction=tune()) %>% 
                  set_engine('mboost',family=Lognormal()) %>%  set_mode('censored regression')
loglog_spec <-    boost_tree(mtry=tune(),min_n=tune(),trees=tune(),tree_depth=tune(),loss_reduction=tune()) %>% 
                  set_engine('mboost',family=Loglog()) %>%  set_mode('censored regression')

## Preprocessing
recipe_other <- recipe(ovarian_survival ~ ., data = train_set)
recipe_boost <- recipe(ovarian_survival ~ ., data=train_set) %>%
    step_impute_knn(all_numeric_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_other(all_nominal_predictors(),threshold=.02) %>%
    step_zv(all_predictors()) 
recipe_other %>% prep() %>% juice()
recipe_boost %>% prep() %>% juice()

## Workflow
surv_wkflow <- workflow() %>%
    add_recipe(recipe_boost) %>%
    add_model(cox_glmnet)
surv_wkflow <- workflow() %>%
    add_recipe(recipe_boost) %>%
    add_model(cox_glmnet,formula=ovarian_survival ~ . + strata(rx))
## Tuning
grid_ctrl <-
   control_grid(
      save_pred = TRUE,
      parallel_over = "resamples",
      save_workflow = FALSE
   )
grid <- surv_wkflow %>% extract_parameter_set_dials() %>% finalize(train_set) %>% 
               grid_space_filling(size = 25)
grid_cox <- surv_wkflow %>% extract_parameter_set_dials() %>% grid_regular(levels = 10) 

plan(multisession,workers=3)
surv_results <- tune_grid(
  surv_wkflow,
  resamples = validate,
  grid = grid_cox,
  metrics = survival_metrics,
  eval_time = evaluation_time_points,
  control = grid_ctrl
)
plan(sequential)
collect_metrics(surv_results) %>% filter(.metric == "concordance_survival") %>% print(n=Inf)


##----------------------------------------------------------------------------------------------------------##
## Machine Learning Building Complaints
##----------------------------------------------------------------------------------------------------------##
set.seed(403)
building_complaints <- modeldatatoo::data_building_complaints()
building_complaints <- building_complaints %>% mutate(disposition_surv = Surv(days_to_disposition, status == "CLOSED"), 
    .keep = "unused")
splits <- initial_validation_split(building_complaints)

complaints_train <- training(splits)
survfit(disposition_surv ~ 1, data = complaints_train) %>% plot()

survreg_spec <- survival_reg(dist=tune()) %>% 
  set_engine("survival") %>% 
  set_mode("censored regression")

rec_other <- recipe(disposition_surv ~ ., data = complaints_train) %>% 
  step_unknown(complaint_priority) %>% 
  step_rm(complaint_category) %>% 
  step_novel(community_board, unit) %>%
  step_other(community_board, unit, threshold = 0.02)

survreg_wflow <- workflow() %>% 
    add_recipe(rec_other) %>% 
    add_model(survreg_spec)

complaints_rset <- validation_set(splits)
survival_metrics <- metric_set(brier_survival_integrated, brier_survival,
                               roc_auc_survival, concordance_survival)
evaluation_time_points <- seq(0, 300, 30)

set.seed(1)
survreg_res <- tune_grid(
  survreg_wflow,
  resamples = complaints_rset,
  grid = grid,
  metrics = survival_metrics,
  eval_time = evaluation_time_points, 
  control = control_resamples(save_pred = TRUE)
)
preds <- collect_predictions(survreg_res)
metrics <- collect_metrics(survreg_res)
metrics %>% print(n=Inf)
collect_metrics(survreg_res) %>% 
  filter(.metric == "roc_auc_survival") %>% 
  ggplot(aes(.eval_time, mean)) + 
  geom_line() + 
  labs(x = "Evaluation Time", y = "Area Under the ROC Curve")
  collect_metrics(survreg_res) %>% 
  filter(.metric == "concordance_survival")

# AFT
rec_unknown <- recipe(disposition_surv ~ ., data = complaints_train) %>% 
  step_unknown(complaint_priority) 
weibull_spec <- boost_tree(mtry=tune(),min_n=tune(),trees=[5,50]) %>% set_engine('mboost',family=Weibull()) %>%  set_mode('censored regression')
weibull_spec %>% translate()
weibull_wflow <- workflow() %>% 
    add_recipe(rec_other) %>% 
    add_model(weibull_spec)
set.seed(1)

weibull_res <- tune_grid(
  weibull_wflow,
  resamples = complaints_rset,
  grid = 25,
  metrics = survival_metrics,
  eval_time = evaluation_time_points, 
  control = control_grid(save_workflow = TRUE)
)
show_best(weibull_res, metric = "brier_survival_integrated", n = 5)
show_best(weibull_res, metric = "concordance_survival", n = 10)
weibull_metrics <- collect_metrics(weibull_res) %>% 
  filter(.metric == "brier_survival_integrated")

# Cox Model
rec_dummies <- rec_unknown %>% 
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

cox_spec <- proportional_hazards(penalty=tune()) %>% set_engine('glmnet') %>%  set_mode('censored regression')
cox_wflow <- workflow() %>% 
    add_recipe(rec_dummies) %>% 
    add_model(cox_spec)

set.seed(1)
cox_res <- tune_grid(
  cox_wflow,
  resamples = cv_folds,
  grid = 25,
  metrics = survival_metrics,
  eval_time = evaluation_time_points, 
  control = control_grid(save_workflow = TRUE)
)
show_best(cox_res, metric = "brier_survival_integrated", n = 5)
show_best(cox_res, metric = "concordance_survival", n = 5)

## Finalize Models

param_best <- select_best(cox_res, metric = "brier_survival_integrated")
last_cox_wflow <- finalize_workflow(cox_wflow, param_best)

set.seed(2)
last_cox_fit <- last_fit(
  last_cox_wflow, 
  split = splits,
  metrics = survival_metrics,
  eval_time = evaluation_time_points, 
)

collect_metrics(last_cox_fit) %>% 
  filter(.metric == "brier_survival_integrated")

## 
brier_val <- collect_metrics(cox_res) %>% 
  filter(.metric == "brier_survival") %>% 
  filter(penalty == param_best$penalty) %>% 
  mutate(Data = "Validation") 
brier_test <- collect_metrics(last_cox_fit) %>% 
  filter(.metric == "brier_survival") %>% 
  mutate(Data = "Testing") %>% 
  rename(mean = .estimate)
bind_rows(brier_val, brier_test) %>% 
  ggplot(aes(.eval_time, mean, col = Data)) + 
  geom_line() + 
  labs(x = "Evaluation Time", y = "Brier Score")


