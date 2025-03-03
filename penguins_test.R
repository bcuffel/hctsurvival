library(tidymodels)
library(ranger)
#> Registered S3 method overwritten by 'tune':
#>   method                   from   
#>   required_pkgs.model_spec parsnip
library(palmerpenguins)

penguins <- penguins %>%
  filter(!is.na(sex)) %>%
  select(-year, -island)

penguins_newdata <- penguins[233:333,-6]


set.seed(123)
penguin_split <- initial_split(penguins, strata = sex)
penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)

rf_spec <- rand_forest() %>%
  set_mode("classification") %>%
  set_engine("ranger")

unfitted_wf <- workflow() %>%
  add_formula(sex ~ .) %>%
  add_model(rf_spec)

penguin_final <- last_fit(unfitted_wf, penguin_split)

collect_metrics(penguin_final)
#> # A tibble: 2 x 4
#>   .metric  .estimator .estimate .config             
#>   <chr>    <chr>          <dbl> <chr>               
#> 1 accuracy binary         0.940 Preprocessor1_Model1
#> 2 roc_auc  binary         0.983 Preprocessor1_Model1


# can predict on this fitted workflow
fitted_wf <- pluck(penguin_final$.workflow, 1)

predict(fitted_wf, new_data = penguins_newdata)
