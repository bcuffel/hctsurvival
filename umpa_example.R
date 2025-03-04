# Load libraries
library(tidymodels)
library(embed)        # For supervised UMAP in recipes
library(survival)     # For survival analysis
library(survminer)    # For Kaplan-Meier plots
library(themis)       # For balancing if needed
library(ggplot2)
library(censored)
library(rsample)      # For resampling and cross-validation
library(tune)         # For tuning and CV
library(yardstick)    # For performance metrics
library(finetune)
library(future)
# ----------------------------------------
# 1. Generate Synthetic Survival Data
# ----------------------------------------

set.seed(123)
n_samples <- 200
n_features <- 50

# Simulate high-dimensional data (e.g., gene expression)
data_matrix <- as.data.frame(matrix(rnorm(n_samples * n_features), ncol = n_features))
colnames(data_matrix) <- paste0("gene_", 1:n_features)

# Simulate survival times and status
survival_time <- rexp(n_samples, rate = 0.1)  # Exponential survival times
censoring <- rbinom(n_samples, 1, 0.7)        # 70% uncensored

# Create risk groups based on survival time for supervised UMAP
/# (e.g., high-risk if survival_time < median)
risk_group <- ifelse(survival_time < median(survival_time), "High", "Low")
risk_group <- cut(survival_time, breaks = quantile(survival_time,probs=seq(0,1,0.25)),include.lowest=TRUE,right=FALSE)
levels(risk_group) <- c("Lowest", "Low","High","Highest")
# Combine into a single dataset
surv_data <- data_matrix %>%
    mutate(hct_survival = Surv(survival_time, censoring==1),
         risk_group = factor(risk_group))  # Categorical target for supervised UMAP

new_surv_data <- surv_data[1:10,] %>% select(-hct_survival,-risk_group)

# Set a seed for reproducibility

set.seed(123)

# Create 5-fold cross-validation with stratification on risk_group
splits <- initial_split(surv_data)
cv_folds <- vfold_cv(surv_data, v = 5, strata = risk_group)

# Inspect the folds
cv_folds

# ----------------------------------------
# 1. Create a Recipe with Supervised UMAP
# ----------------------------------------

# Create a recipe using risk_group to guide UMAP
umap_recipe <- recipe(hct_survival ~ ., data = training(splits)) %>%
    update_role(risk_group, new_role = "id") %>%  # Ignore at prediction time
      update_role_requirements(role = "id", bake = FALSE) %>%  # Ensure it's not required at bake()
    step_umap(all_predictors(), outcome = vars(risk_group), num_comp = tune(),
              neighbors=tune(),min_dist=tune()) %>%
  step_normalize(all_numeric_predictors())

# ----------------------------------------
# Specify cox model
# ----------------------------------------

# Cox Proportional Hazards Model
cox_model <- proportional_hazards(mode = "censored regression") %>%
  set_engine("survival")
cox_mboost <- boost_tree(mtry=5,min_n=tune(),trees=tune(),tree_depth = tune(),loss_reduction = .1) %>%
    set_engine('mboost') %>% set_mode('censored regression')

# ----------------------------------------
# create work flow
# ----------------------------------------

# Workflow
surv_workflow <- workflow() %>%
  add_recipe(umap_recipe) %>%
  add_model(cox_mboost)


# Define survival metrics
survival_metrics <- metric_set(concordance_survival,brier_survival_integrated, brier_survival,
                               roc_auc_survival)
evaluation_time_points <- seq(0,60, by=10)

# Cross-validation using tune_grid
plan(multisession,workers=3)
grid <- surv_workflow %>% extract_parameter_set_dials() %>% grid_space_filling(grid = 25)
grid
cv_results <- surv_workflow %>%
  tune_grid(
    resamples = cv_folds,
    metrics = survival_metrics,
    control = control_grid(),
    eval_time = evaluation_time_points,
    grid = grid
  )
plan(sequential)
# Collect and view metrics
collect_metrics(cv_results)
cv_results

# Select best hyperparameters
best_params <- select_best(cv_results, metric = "concordance_survival")

# Finalize the workflow with best parameters
last_surv_workflow <- finalize_workflow(surv_workflow, best_params)

# Fit the model directly on training data
last_surv_fit <- fit(last_surv_workflow, data = training(splits))

# Extract the trained preprocessor (recipe)
final_recipe <- last_surv_fit %>%
  extract_preprocessor() %>%
  prep(training = training(splits), retain = TRUE)


# Remove survival outcome columns from new data
new_surv_data <- surv_data %>%
  select(-hct_survival, -risk_group)

# Apply trained transformations (UMAP + normalization) to new data
new_data_processed <- bake(final_recipe, new_data = new_surv_data)
new_data_processed <- as.data.frame(new_data_processed)
# Predict risk scores (linear predictor)

new_risk_scores <- predict(last_surv_fit, new_data_processed, type = "linear_pred")

# View predictions
head(new_risk_scores)



## -- Debugging
# Step 1: Select best hyperparameters
best_params <- select_best(cv_results, metric = "concordance_survival")

# Step 2: Create a fresh workflow from scratch (without retaining old steps)
last_surv_workflow <- workflow() %>%
  add_recipe(umap_recipe) %>%  # Ensure UMAP is included from the start
  add_model(cox_model)

# Step 3: Finalize the workflow with best hyperparameters
last_surv_workflow <- finalize_workflow(last_surv_workflow, best_params)


# Step 4: Prepare the final recipe
final_recipe <- last_surv_workflow %>%
  extract_preprocessor() %>%
  prep(training = training(splits), retain = TRUE)

# Step 5: Apply preprocessing to the training data
train_processed <- bake(final_recipe, new_data = training(splits))

# Step 6: Refit the model with the correct dataset
last_surv_fit <- last_surv_workflow %>%
  fit(data = train_processed)
