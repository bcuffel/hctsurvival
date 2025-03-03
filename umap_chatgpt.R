# Load required libraries
library(tidymodels)  # Core modeling framework
library(embed)        # For supervised UMAP in recipes
library(survival)     # For survival analysis
library(censored)     # For Cox model in tidymodels
library(umap)         # Core UMAP library

# -------------------------
# 1. Generate Synthetic Survival Data
# -------------------------

set.seed(123)
n_samples <- 200
n_features <- 50

# Simulate high-dimensional gene expression data
gene_data <- as.data.frame(matrix(rnorm(n_samples * n_features), ncol = n_features))
colnames(gene_data) <- paste0("gene_", 1:n_features)

# Simulate survival times and event status
survival_time <- rexp(n_samples, rate = 0.1)  # Exponential survival times
censoring <- rbinom(n_samples, 1, 0.7)        # 70% event occurrence

# Create a categorical risk group (needed for supervised UMAP)
risk_group <- cut(survival_time, breaks = quantile(survival_time, probs = seq(0,1,0.25)), 
                  include.lowest = TRUE, right = FALSE)
levels(risk_group) <- c("Lowest", "Low", "High", "Highest")

# Combine into a single dataset
surv_data <- gene_data %>%
  mutate(surv_obj = Surv(survival_time, censoring == 1),  # Survival object
         risk_group = factor(risk_group))  # Target for UMAP supervision

# -------------------------
# 2. Data Splitting
# -------------------------

# Initial train-test split (80% training, 20% testing)
set.seed(123)
splits <- initial_split(surv_data, prop = 0.8, strata = risk_group)
train_data <- training(splits)
test_data <- testing(splits)

# 5-fold cross-validation on the training set
cv_folds <- vfold_cv(train_data, v = 5, strata = risk_group)

# -------------------------
# 3. Define the Recipe with Supervised UMAP
# -------------------------

umap_recipe <- recipe(surv_obj ~ ., data = train_data) %>%
  update_role(risk_group, new_role = "id") %>%  # Mark risk_group as non-predictor
  update_role_requirements(role = "id", bake = FALSE) %>%  # Ensure it's not needed for new data
  step_umap(all_numeric_predictors(), outcome = vars(risk_group), num_comp = 5, neighbors = 15, min_dist = 0.01) %>%
  step_normalize(all_numeric_predictors())  # Normalize UMAP embeddings

# -------------------------
# 4. Define the Cox Model
# -------------------------

#cox_model <- proportional_hazards(mode = "censored regression") %>%
#  set_engine("survival")
cox_model <- boost_tree(mtry=5,min_n=tune(),trees=tune(),tree_depth = tune(),loss_reduction = .1) %>%
    set_engine('mboost') %>% set_mode('censored regression')

# -------------------------
# 5. Define Workflow
# -------------------------

surv_workflow <- workflow() %>%
  add_recipe(umap_recipe) %>%
  add_model(cox_model)

# -------------------------
# 6. Cross-Validation
# -------------------------

set.seed(123)
cv_results <- fit_resamples(
  surv_workflow,
  resamples = cv_folds,
  metrics = metric_set(concordance_survival),
  control = control_resamples(save_pred = TRUE)
)

# View cross-validation results
collect_metrics(cv_results)

# -------------------------
# 7. Train the Final Model
# -------------------------

# Fit the model on the full training set
last_surv_fit <- fit(surv_workflow, data = train_data)

# -------------------------
# 8. Apply the Trained Model to New Data
# -------------------------

# Extract the trained recipe for preprocessing new data
final_recipe <- last_surv_fit %>%
  extract_preprocessor() %>%
  prep(training = train_data, retain = TRUE)

# Remove survival outcomes before preprocessing test data
new_test_data <- test_data %>%
  select(-surv_obj, -risk_group)

# Apply trained UMAP + normalization to test data
new_test_processed <- bake(final_recipe, new_data = new_test_data)

# Predict risk scores
new_risk_scores <- predict(last_surv_fit, test_data %>% select(-surv_obj, -risk_group), type = "linear_pred")


#new_risk_scores <- predict(last_surv_fit, new_test_processed, type = "linear_pred")

# View predictions
head(new_risk_scores)

# -------------------------
# 9. Predict Survival Probabilities (Optional)
# -------------------------

# Define time grid for survival predictions
time_grid <- seq(0, 60, by = 10)

# Predict survival probabilities
new_survival_probs <- predict(last_surv_fit, new_test_processed, type = "survival", eval_time = time_grid)

# View predictions
print(new_survival_probs)
