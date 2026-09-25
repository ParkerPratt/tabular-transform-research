library(tidyverse)
library(readxl)
library(broom)

frozen_model = read.csv("results/frozen_model_coefficients.csv")

holdout_energy = read.csv("results/holdout_yeo_metrics.csv")
holdout_curvature = read.csv("results/holdout_yeo_curvature.csv")


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


holdout_results_fixed = read_excel("results/holdout_yeo_independent.xlsx")

Averaged_fixed = holdout_results_fixed %>%
  group_by(transform, strength, n, 
           observed_representation, 
           signal_representation
  ) %>%
  summarize(
    mean_r2 = mean(r2_test),
    .groups = "drop"
  )

HoldoutDegrade_fixed = Averaged_fixed %>%
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

HoldoutDegrade_fixed = HoldoutDegrade_fixed %>%
  left_join(
    holdout_energy,
    by = c("transform", "strength")
  ) %>%
  left_join(
    holdout_curvature,
    by = c("transform", "strength")
  ) 

HoldoutDegrade_fixed = HoldoutDegrade_fixed %>%
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

ForwardHoldout_fixed = HoldoutDegrade_fixed %>%
  filter(direction == "forward")

InverseHoldout_fixed = HoldoutDegrade_fixed %>%
  filter(direction == "inverse")

ForwardHoldout_fixed$pred_base =
  predict_frozen(ForwardHoldout_fixed, frozen_model, "forward_base")

ForwardHoldout_fixed$pred_energy =
  predict_frozen(ForwardHoldout_fixed, frozen_model, "forward_energy")

ForwardHoldout_fixed$pred_excess =
  predict_frozen(ForwardHoldout_fixed, frozen_model, "forward_excess")

ForwardHoldout_fixed$pred_curve =
  predict_frozen(ForwardHoldout_fixed, frozen_model, "forward_curve")

InverseHoldout_fixed$pred_base =
  predict_frozen(InverseHoldout_fixed, frozen_model, "inverse_base")

InverseHoldout_fixed$pred_energy =
  predict_frozen(InverseHoldout_fixed, frozen_model, "inverse_energy")

InverseHoldout_fixed$pred_excess =
  predict_frozen(InverseHoldout_fixed, frozen_model, "inverse_excess")

InverseHoldout_fixed$pred_curve =
  predict_frozen(InverseHoldout_fixed, frozen_model, "inverse_curve")

HoldoutPredictions_fixed = bind_rows(
  ForwardHoldout_fixed,
  InverseHoldout_fixed
)

model_levels <- c("base", "energy", "excess", "curve")
model_labels <- c(
  "Nonlinearity",
  "E[2] + E[3]",
  "E[2] + E[phantom() >= 3]",
  "Curvature"
)
HoldoutComparison_fixed = HoldoutPredictions_fixed %>%
  pivot_longer(
    cols = starts_with("pred_"),
    names_to = "model",
    values_to = "prediction"
  ) %>%
  mutate(
    model = str_remove(model, "pred_"),
    model = factor(model, levels = model_levels, labels = model_labels),
    direction = factor(
      direction,
      levels = c("forward", "inverse"),
      labels = c("Forward~map", "Inverse~map")
    )
  )

HoldoutMetrics_fixed = HoldoutComparison_fixed %>%
  group_by(direction, model) %>%
  summarize(
    rmse = sqrt(mean((degradation - prediction)^2)),
    mae = mean(abs(degradation - prediction)),
    r2 = 1 - sum((degradation - prediction)^2) /
      sum((degradation - mean(degradation))^2),
    correlation = cor(degradation, prediction),
    .groups = "drop"
  )

HoldoutMetrics_fixed

holdoutpred = ggplot(
  HoldoutComparison_fixed,
  aes(x = degradation, y = prediction)
) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  facet_grid(direction ~ model, scales = "free", labeller = label_parsed) +
  labs(
    x = expression("Actual degradation (" * Delta * R^2 * ")"),
    y = expression("Predicted degradation (" * Delta * R^2 * ")")
  ) +
  theme_minimal()
holdoutpred

ggsave(
  "figures/holdout_pred.pdf",
  holdoutpred,
  width = 5.5,
  height = 3.6,
  units = "in",
  device = cairo_pdf
)
