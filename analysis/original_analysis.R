library(tidyverse)
library(readxl)
library(broom)

files = list.files(
  path = "../results",
  pattern = "xgb_monotone",
  full.names = TRUE
)

raw = bind_rows(lapply(files, function(f) {
  read_excel(f)
}))

baseline <- raw %>%
  filter(
    observed_representation == "transformed",
    signal_representation == "transformed"
  )

baseline %>%
  group_by(model) %>%
  summarise(
    mean_r2 = mean(r2_test, na.rm = TRUE),
    sd_r2 = sd(r2_test, na.rm = TRUE),
    reps = n(),
    .groups = "drop"
  )
baseline %>%
  group_by(model, transform, strength, n) %>%
  summarise(
    mean_r2 = mean(r2_test, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = strength,
    y = mean_r2,
    color = transform
  )) +
  geom_point() +
  geom_smooth(method = "lm",se=FALSE)+
  facet_wrap(~ model+n)

hermite = read.csv("../results/hermite_metrics.csv")

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
  )

ObservedDegrade = analysis %>%
  filter(signal_representation == "transformed") %>%
  select(
    n, model, transform, strength,
    observed_representation, mean_r2,
    starts_with("c"),
    starts_with("energy"),
    higher_order_energy,
    weighted_higher_order_energy
  ) %>%
  pivot_wider(
    names_from = observed_representation,
    values_from = mean_r2
  ) %>%
  mutate(
    degradation = transformed - base
  )

fit = lm(
  degradation ~ log(n) + energy2 + energy3 + energy4 + energy5,
  data = ObservedDegrade %>%
    filter(model == "ann")
)

summary(fit)

fitlinear = lm(
  degradation ~ log(n) + energy1,
  data = ObservedDegrade %>%
    filter(model == "ann")
)
summary(fitlinear)

