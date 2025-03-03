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
risk_group <- cut(survival_time, breaks = quantile(survival_time,probs=seq(0,1,0.25)),include.lowest=TRUE,right=FALSE)
levels(risk_group) <- c("Lowest", "Low","High","Highest")

# Combine into a single dataset
surv_data <- data_matrix %>%
    mutate(hct_survival = Surv(survival_time, censoring==1),
           risk_group = factor(risk_group))  # Categorical target for supervised UMAP
# Set a seed for reproducibility

set.seed(123)
splits <- initial_split(surv_data)
train_data <- training(splits)
test_data <-  testing(splits)
cv_folds <- vfold_cv(surv_data, v = 5, strata = risk_group)

# ----------------------------------------
#  Create a Recipe with Supervised UMAP
# ----------------------------------------
umap_recipe <- recipe(hct_survival ~ ., data = train_data) %>%
  update_role(risk_group, new_role = "id") %>%  # Mark risk_group as non-predictor
  update_role_requirements(role = "id", bake = FALSE) %>%  # Ensure it's not needed for new data
  step_umap(all_numeric_predictors(), outcome = vars(risk_group), num_comp = 5, neighbors = 15, min_dist = 0.01) %>%
  step_normalize(all_numeric_predictors())  # Normalize UMAP embeddings

# ----------------------------------------
# Specify cox model
# ----------------------------------------
cox_model <- boost_tree(mtry=5,min_n=40,trees=2000,tree_depth = 8,loss_reduction = .1) %>%
    set_engine('mboost') %>% set_mode('censored regression')

# ----------------------------------------
# create work flow
# ----------------------------------------

# Workflow
surv_workflow <- workflow() %>%
  add_recipe(umap_recipe) %>%
  add_model(cox_model)

#----------------------------------------
# Cross validation 
#----------------------------------------

set.seed(123)
cv_results <- fit_resamples(
  surv_workflow,
  resamples = cv_folds,
  metrics = metric_set(concordance_survival),
  control = control_resamples(save_pred = TRUE)
)

# Collect and view metrics
collect_metrics(cv_results)

# Fit the model on the full training set
last_surv_fit <- fit(surv_workflow, data = train_data)

# Verify that the model now expects UMAP embeddings
expected_features <- last_surv_fit %>%
  extract_mold() %>%
  pluck("predictors") %>%
  colnames()

print(expected_features)  # Should output: "UMAP1", "UMAP2", "UMAP3", "UMAP4", "UMAP5"

# -------------------------
# 8. Apply the Trained Model to New Data
# -------------------------

# Use the trained workflow directly for prediction
new_risk_scores <- predict(last_surv_fit, test_data %>% select(-surv_obj, -risk_group), type = "linear_pred")

# View predictions
head(new_risk_scores)



# Select best hyperparameters
best_params <- select_best(cv_results, metric = "concordance_survival")

# Finalize the workflow with best parameters
last_surv_workflow <- finalize_workflow(surv_workflow, best_params)

# Fit the model directly on training data
last_surv_fit <- fit(last_surv_workflow, data = training(splits))

expected_features <- last_surv_fit %>%
  extract_mold() %>%
  pluck("predictors") %>%
  colnames()

print(expected_features)  # Should output: "UMAP1", "UMAP2", "UMAP3", "UMAP4", "UMAP5"
new_risk_scores <- predict(last_surv_fit, test_data %>% select(-hct_survival, -risk_group), type = "linear_pred")

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
