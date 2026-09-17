library(tidyverse)
library(readxl)
library(broom)

frozen_model = read.csv("../results/frozen_model_coefficients.csv")

holdout_energy = read.csv("../results/holdout_yeo_metrics.csv")
holdout_curvature = read.csv("../results/holdout_yeo_curvature.csv")

holdout_results = read_excel("../results/holdout_yeo.xlsx")

Averaged = holdout_results %>%
  group_by(transform, strength, n, 
           observed_representation, 
           signal_representation
           ) %>%
  summarize(
    mean_r2 = mean(r2_test),
    .groups = "drop"
  )

HoldoutDegrade = Averaged %>%
  pivot_wider(
    names_from = observed_representation,
    values_from = mean_r2
  ) %>%
  mutate(
    direction = if_else(
      signal_representation == "transformed",
      "forward",
      "inverse"
    ),
    
    degradation = if_else(
      signal_representation == "transformed",
      transformed - base,
      base - transformed
    )
  ) 

HoldoutDegrade = HoldoutDegrade %>%
  left_join(
    holdout_energy,
    by = c("transform", "strength")
  ) %>%
  left_join(
    holdout_curvature,
    by = c("transform", "strength")
  ) 

HoldoutDegrade = HoldoutDegrade %>%
  mutate(
    nonlinear_energy = 1 - energy1,
    energy3plus = 1-energy1-energy2
  ) %>%
  mutate(
    inv_nonlinear_energy = 1 - inv_energy1,
    inv_energy3plus = 1 - inv_energy1 - inv_energy2
  )

predict_frozen = function(data, coef_table, model_name) {
  
  coefs = coef_table %>%
    filter(model == model_name)
  
  prediction = rep(0, nrow(data))
  
  for (i in 1:nrow(coefs)) {
    
    term = coefs$term[i]
    beta = coefs$estimate[i]
    
    if (term == "(Intercept)") {
      
      prediction = prediction + beta
      
    } else if (term == "log(n)") {
      
      prediction = prediction + beta * log(data$n)
      
    } else if (str_detect(term, ":")) {
      
      pieces = str_split(term, ":", simplify = TRUE)
      
      x1 = if (pieces[1] == "log(n)") {
        log(data$n)
      } else {
        data[[pieces[1]]]
      }
      
      x2 = if (pieces[2] == "log(n)") {
        log(data$n)
      } else {
        data[[pieces[2]]]
      }
      
      prediction = prediction + beta * x1 * x2
      
    } else {
      
      prediction = prediction + beta * data[[term]]
      
    }
  }
  
  prediction
}

ForwardHoldout = HoldoutDegrade %>%
  filter(direction == "forward")

InverseHoldout = HoldoutDegrade %>%
  filter(direction == "inverse")

ForwardHoldout$pred_base =
  predict_frozen(ForwardHoldout, frozen_model, "forward_base")

ForwardHoldout$pred_energy =
  predict_frozen(ForwardHoldout, frozen_model, "forward_energy")

ForwardHoldout$pred_excess =
  predict_frozen(ForwardHoldout, frozen_model, "forward_excess")

ForwardHoldout$pred_curve =
  predict_frozen(ForwardHoldout, frozen_model, "forward_curve")

InverseHoldout$pred_base =
  predict_frozen(InverseHoldout, frozen_model, "inverse_base")

InverseHoldout$pred_energy =
  predict_frozen(InverseHoldout, frozen_model, "inverse_energy")

InverseHoldout$pred_excess =
  predict_frozen(InverseHoldout, frozen_model, "inverse_excess")

InverseHoldout$pred_curve =
  predict_frozen(InverseHoldout, frozen_model, "inverse_curve")

HoldoutPredictions = bind_rows(
  ForwardHoldout,
  InverseHoldout
)

HoldoutComparison = HoldoutPredictions %>%
  pivot_longer(
    cols = starts_with("pred_"),
    names_to = "model",
    values_to = "prediction"
  ) %>%
  mutate(
    model = str_remove(model, "pred_")
  )

HoldoutMetrics = HoldoutComparison %>%
  group_by(direction, model) %>%
  summarize(
    rmse = sqrt(mean((degradation - prediction)^2)),
    mae = mean(abs(degradation - prediction)),
    r2 = 1 - sum((degradation - prediction)^2) /
      sum((degradation - mean(degradation))^2),
    correlation = cor(degradation, prediction),
    .groups = "drop"
  )

HoldoutMetrics

ggplot(
  HoldoutComparison,
  aes(x = degradation, y = prediction)
) +
  geom_point() +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  facet_grid(direction ~ model, scales = "free") +
  labs(
    x = "Actual degradation",
    y = "Predicted degradation"
  ) +
  theme_minimal()


