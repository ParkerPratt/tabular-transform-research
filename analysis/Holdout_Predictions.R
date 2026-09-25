library(tidyverse)
library(readxl)
library(broom)

files = list.files(
  path = "results",
  pattern = "matched_nonlinearity_transform",
  full.names = TRUE
)

raw = bind_rows(lapply(files, function(f) {
  read_excel(f)
}))

hermite = read.csv("results/matched_hermite_metrics.csv")
curvature = read.csv("results/matched_curvature_metrics.csv")
inverse = read.csv("results/inverse_polynomial_metrics.csv")

#Forward Direction
analysis = raw %>%
  group_by(
    n, transform, strength,
    signal_representation,
    observed_representation,
    model
  ) %>%
  summarize(
    mean_r2 = mean(r2_test),
    .groups = "drop"
  ) %>%
  left_join(
    hermite,
    by = c("transform", "strength")
  )%>%
  left_join(
    curvature,
    by = c("transform", "strength")
  )


ObservedDegrade = analysis %>%
  filter(signal_representation == "transformed") %>%
  select(
    n, model, transform, strength,
    observed_representation, mean_r2,
    starts_with("energy"),
    curvature
  ) %>%
  pivot_wider(
    names_from = observed_representation,
    values_from = mean_r2
  ) %>%
  mutate(
    degradation = transformed - base
  )
ObservedDegrade = ObservedDegrade %>%
  mutate(
    nonlinear_energy = 1 - energy1,
    energy3plus = 1-energy1-energy2
  )

forward_basefit = lm(
  degradation ~ log(n)*nonlinear_energy,
  data = ObservedDegrade %>%
    filter(model == "ann")
)
summary(forward_basefit)


forward_energyfit = lm(
  degradation ~ log(n) * (energy2 + energy3),
  data = ObservedDegrade %>%
    filter(model == "ann")
)
summary(forward_energyfit)

forward_excessenergyfit = lm(
  degradation ~ log(n) * (energy2 + energy3plus),
  data = ObservedDegrade %>%
    filter(model == "ann")
)
summary(forward_excessenergyfit)

forward_curvefit = lm(
  degradation ~ log(n) * (curvature + nonlinear_energy),
  data = ObservedDegrade %>%
    filter(model == "ann")
)
summary(forward_curvefit)

#Inverse
analysis = raw %>%
  group_by(
    n, transform, strength,
    signal_representation,
    observed_representation,
    model
  ) %>%
  summarize(
    mean_r2 = mean(r2_test),
    .groups = "drop"
  ) %>%
  left_join(
    inverse,
    by = c("transform", "strength")
  )%>%
  left_join(
    curvature,
    by = c("transform", "strength")
  )

BaseDegrade = analysis %>%
  filter(signal_representation == "base") %>%
  select(
    n, model, transform, strength,
    observed_representation, mean_r2,
    starts_with("inv_energy"),
    curvature
  ) %>%
  pivot_wider(
    names_from = observed_representation,
    values_from = mean_r2
  ) %>%
  mutate(
    degradation = base - transformed
  )
BaseDegrade = BaseDegrade %>%
  mutate(
    inv_nonlinear_energy = 1 - inv_energy1,
    inv_energy3plus = 1 - inv_energy1 - inv_energy2
  )

inverse_basefit = lm(
  degradation ~ log(n)*inv_nonlinear_energy,
  data = BaseDegrade %>%
    filter(model == "ann")
)
summary(inverse_basefit)


inverse_energyfit = lm(
  degradation ~ log(n) * (inv_energy2 + inv_energy3),
  data = BaseDegrade %>%
    filter(model == "ann")
)
summary(inverse_energyfit)

inverse_excessenergyfit = lm(
  degradation ~ log(n) * (inv_energy2 + inv_energy3plus),
  data = BaseDegrade %>%
    filter(model == "ann")
)
summary(inverse_excessenergyfit)

inverse_curvefit = lm(
  degradation ~ log(n) * (curvature + inv_nonlinear_energy),
  data = BaseDegrade %>%
    filter(model == "ann")
)
summary(inverse_curvefit)

frozen_models = list(
  forward_base = forward_basefit,
  forward_energy = forward_energyfit,
  forward_excess = forward_excessenergyfit,
  forward_curve = forward_curvefit,
  inverse_base = inverse_basefit,
  inverse_energy = inverse_energyfit,
  inverse_excess = inverse_excessenergyfit,
  inverse_curve = inverse_curvefit
)

frozen_coefficients = bind_rows(
  lapply(names(frozen_models), function(name) {
    tidy(frozen_models[[name]]) %>%
      mutate(model = name, .before = 1)
  })
)

write.csv(
  frozen_coefficients,
  "results/frozen_model_coefficients.csv",
  row.names = FALSE
)

frozen_model_fits = bind_rows(
  lapply(names(frozen_models), function(name) {
    glance(frozen_models[[name]]) %>%
      mutate(model = name, .before = 1)
  })
)

write.csv(
  frozen_model_fits,
  "results/frozen_model_fits.csv",
  row.names = FALSE
)




