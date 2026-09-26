library(tidyverse)
library(readxl)

ActivePredRaw = read_excel("results/four_condition_diagnostics_modern.xlsx")

ActivePred = ActivePredRaw %>%
  filter(n_inactive_transformed == 0) %>%
  group_by(
    n_active_transformed,
    transform,
    observed_representation,
    signal_representation,
    model
  ) %>%
  summarize(
    mean_r2 = mean(r2_test, na.rm = TRUE),
    .groups = "drop"
  )

ActivePredPlot = ActivePred %>%
  filter(
    model == "ann",
    (observed_representation == "base" &
       signal_representation == "transformed") |
      (observed_representation == "transformed" &
         signal_representation == "base")
  ) %>%
  mutate(
    direction = case_when(
      observed_representation == "base" &
        signal_representation == "transformed" ~ "Forward map",
      
      observed_representation == "transformed" &
        signal_representation == "base" ~ "Inverse map"
    ),
    n_mismatched_active = n_active_transformed
  )

BaselinePlot = ActivePredRaw %>%
  filter(
    n_inactive_transformed == 0,
    model == "ann",
    (observed_representation == "base" &
       signal_representation == "base") |
      (observed_representation == "transformed" &
         signal_representation == "transformed")
  ) %>%
  mutate(
    direction = case_when(
      observed_representation == "base" &
        signal_representation == "base" ~ "Forward map",
      
      observed_representation == "transformed" &
        signal_representation == "transformed" ~ "Inverse map"
    )
  ) %>%
  group_by(transform, direction) %>%
  summarize(
    mean_r2 = mean(r2_test, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    n_mismatched_active = 0
  )

ActivePredPlot = ActivePredPlot %>%
  select(
    transform,
    direction,
    n_mismatched_active,
    mean_r2
  ) %>%
  bind_rows(BaselinePlot)


p_active = ggplot(
  ActivePredPlot,
  aes(
    x = n_mismatched_active,
    y = mean_r2,
    colour = transform,
    shape = transform
  )
) +
  geom_point(size = 1.7) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    linewidth = 0.65
  ) +
  facet_wrap(~ direction, nrow = 1) +
  scale_x_continuous(breaks = 0:10) +
  labs(
    x = "Number of mismatched active predictors",
    y = expression("Test " * R^2),
    colour = NULL,
    shape = NULL
  ) +
  theme_classic(base_size = 9.5) +
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 9.5),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5)
  )
p_active

ggsave(
  "figures/active_predictor_degradation.pdf",
  p_active,
  width = 5.5,
  height = 2.8,
  units = "in",
  device = cairo_pdf
)
#Linear test
LinearFits = ActivePredPlot %>%
  group_by(direction, transform) %>%
  group_modify(~{
    fit = lm(mean_r2 ~ n_mismatched_active, data = .x)
    
    tibble(
      slope = coef(fit)[2],
      linear_r2 = summary(fit)$r.squared
    )
  }) %>%
  ungroup()

LinearFits
##Inactive predictor transformation
InactivePaired = ActivePredRaw %>%
  filter(
    n_active_transformed == 10,
    n_inactive_transformed %in% c(0, 40)
  ) %>%
  select(
    rep,
    transform,
    strength,
    observed_representation,
    signal_representation,
    model,
    n_active_transformed,
    n_inactive_transformed,
    r2_test
  ) %>%
  pivot_wider(
    names_from = n_inactive_transformed,
    values_from = r2_test,
    names_prefix = "inactive_"
  ) %>%
  mutate(
    r2_difference = inactive_40 - inactive_0
  )
InactivePaired %>%
  summarize(
    mean_difference = mean(r2_difference),
    sd_difference = sd(r2_difference),
    se_difference = sd(r2_difference) / sqrt(n()),
    n = n()
  )

##Quantile
Quantile = read_excel(
  "results/matched_nonlinearity_quantile_buckets_n500_p50_a10.xlsx",
  sheet = "paired_summary"
)

QuantilePlot = Quantile %>%
  filter(
    model == "ann",
    bucket_by == "signal",
    bucket != "overall",
    nonlinearity_level == 0.30
  ) %>%
  mutate(
    direction = case_when(
      comparison == "BT-TT" ~ "Forward map",
      comparison == "TB-BB" ~ "Inverse map"
    ),
    bucket = factor(
      bucket,
      levels = c("low", "middle", "high"),
      labels = c("Low tail", "Middle", "High tail")
    )
  )
p_quantile = ggplot(
  QuantilePlot,
  aes(
    x = bucket,
    y = delta_signal_mse_mean,
    colour = transform,
    shape = transform
  )
) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.4
  ) +
  geom_errorbar(
    aes(
      ymin = delta_signal_mse_ci95_low,
      ymax = delta_signal_mse_ci95_high
    ),
    width = 0.08,
    linewidth = 0.45,
    position = position_dodge(width = 0.45)
  ) +
  geom_point(
    size = 2,
    position = position_dodge(width = 0.45)
  ) +
  facet_wrap(~ direction, nrow = 1) +
  labs(
    x = "Signal region",
    y = expression(Delta * " signal MSE"),
    colour = NULL,
    shape = NULL
  ) +
  theme_classic(base_size = 9.5) +
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 9.5),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5)
  )

p_quantile

ggsave(
  "figures/quantile_signal_error.pdf",
  p_quantile,
  width = 5.5,
  height = 2.8,
  units = "in",
  device = cairo_pdf
)

QuantilePaired = read_excel(
  "results/matched_nonlinearity_quantile_buckets_n500_p50_a10.xlsx",
  sheet = "paired"
)

SSEContribution = QuantilePaired %>%
  filter(
    model == "ann",
    bucket_by == "signal",
    bucket != "overall",
    nonlinearity_level == 0.30
  ) %>%
  mutate(
    direction = case_when(
      comparison == "BT-TT" ~ "Forward map",
      comparison == "TB-BB" ~ "Inverse map"
    )
  ) %>%
  group_by(rep, transform, direction) %>%
  summarize(
    low_sse = sum(delta_sse[bucket == "low"]),
    middle_sse = sum(delta_sse[bucket == "middle"]),
    high_sse = sum(delta_sse[bucket == "high"]),
    
    total_sse = low_sse + middle_sse + high_sse,
    
    tail_sse = low_sse + high_sse,
    
    tail_share = tail_sse / total_sse,
    middle_share = middle_sse / total_sse,
    
    .groups = "drop"
  )
SSESummary = SSEContribution %>%
  group_by(transform, direction) %>%
  summarize(
    mean_tail_share = mean(tail_share),
    se_tail_share = sd(tail_share) / sqrt(n()),
    
    mean_middle_share = mean(middle_share),
    se_middle_share = sd(middle_share) / sqrt(n()),
    
    .groups = "drop"
  )

SSESummary

##Matched Linearity
Matchedfiles = list.files(
  path = "results",
  pattern = "matched_nonlinearity_transform",
  full.names = TRUE
)

Matchedraw = bind_rows(lapply(Matchedfiles, function(f) {
  read_excel(f)
}))

StrengthLookup = Matchedraw %>%
  distinct(transform, strength) %>%
  group_by(transform) %>%
  arrange(strength, .by_group = TRUE) %>%
  mutate(
    nonlinearity = c(0.05, 0.1, 0.15, 0.2, 0.3)
  ) %>%
  ungroup()

MatchedPlot = Matchedraw %>%
  group_by(
    n, transform, model,
    strength,
    observed_representation,
    signal_representation
  ) %>%
  summarize(
    mean_r2 = mean(r2_test, na.rm = TRUE),
    se_r2 = sd(r2_test, na.rm = TRUE) /
      sqrt(sum(!is.na(r2_test))),
    lower = mean_r2 - 1.96 * se_r2,
    upper = mean_r2 + 1.96 * se_r2,
    .groups = "drop"
  ) %>%
  left_join(
    StrengthLookup,
    by = c("transform", "strength")
  )


LearningCurvePlot = MatchedPlot %>%
  filter(
    model == "ann",
    near(nonlinearity, 0.1) | near(nonlinearity, 0.3),
    (observed_representation == "base" &
       signal_representation == "transformed") |
      (observed_representation == "transformed" &
         signal_representation == "base")
  ) %>%
  mutate(
    direction = case_when(
      observed_representation == "base" &
        signal_representation == "transformed" ~ "Forward map",
      
      observed_representation == "transformed" &
        signal_representation == "base" ~ "Inverse map"
    ),
    
    nonlinearity_panel = case_when(
      near(nonlinearity, 0.1) ~ "Nonlinearity = 0.1",
      near(nonlinearity, 0.3) ~ "Nonlinearity = 0.3"
    )
  )

ForwardControl = MatchedPlot %>%
  filter(
    model == "ann",
    near(nonlinearity, 0.1) | near(nonlinearity, 0.3),
    observed_representation == "transformed",
    signal_representation == "transformed"
  ) %>%
  mutate(
    direction = "Forward map",
    
    nonlinearity_panel = case_when(
      near(nonlinearity, 0.1) ~ "Nonlinearity = 0.1",
      near(nonlinearity, 0.3) ~ "Nonlinearity = 0.3"
    )
  )

InverseControl = Matchedraw %>%
  filter(
    model == "ann",
    observed_representation == "base",
    signal_representation == "base"
  ) %>%
  group_by(n, rep) %>%
  summarize(
    rep_r2 = mean(r2_test, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(n) %>%
  summarize(
    mean_r2 = mean(rep_r2, na.rm = TRUE),
    se_r2 = sd(rep_r2, na.rm = TRUE) /
      sqrt(sum(!is.na(rep_r2))),
    lower = mean_r2 - 1.96 * se_r2,
    upper = mean_r2 + 1.96 * se_r2,
    .groups = "drop"
  ) %>%
  crossing(
    direction = "Inverse map",
    nonlinearity_panel = c(
      "Nonlinearity = 0.1",
      "Nonlinearity = 0.3"
    )
  )

p_learning = ggplot(
  LearningCurvePlot,
  aes(
    x = n,
    y = mean_r2,
    colour = transform,
    shape = transform,
    group = transform
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  
  # Forward-map TT matched controls
  geom_line(
    data = ForwardControl,
    aes(
      x = n,
      y = mean_r2,
      colour = transform,
      group = transform
    ),
    inherit.aes = FALSE,
    linetype = "dotted",
    linewidth = 0.7
  ) +
  
  # Inverse-map BB matched control
  geom_line(
    data = InverseControl,
    aes(
      x = n,
      y = mean_r2
    ),
    inherit.aes = FALSE,
    colour = "black",
    linetype = "dashed",
    linewidth = 0.7
  ) +
  
  facet_grid(
    direction ~ nonlinearity_panel
  ) +
  
  scale_x_log10(
    breaks = sort(unique(LearningCurvePlot$n)),
    labels = sort(unique(LearningCurvePlot$n))
  ) +
  
  labs(
    x = "Sample size",
    y = expression("Test " * R^2),
    colour = NULL,
    shape = NULL
  ) +
  
  theme_classic(base_size = 9.5) +
  
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 9.5
    ),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p_learning

ggsave(
  "figures/nonlinearity_r2_n.pdf",
  p_learning,
  width = 5.5,
  height = 3,
  units = "in",
  device = cairo_pdf
)

p_learning_ci = ggplot(
  LearningCurvePlot,
  aes(
    x = n,
    y = mean_r2,
    colour = transform,
    shape = transform,
    group = transform
  )
) +
  
  # mismatch curves
  geom_line(linewidth = 0.8) +
  
  # mismatch CIs
  geom_linerange(
    aes(
      ymin = lower,
      ymax = upper
    ),
    linewidth = 0.5,
    alpha = 0.75
  ) +
  
  geom_point(size = 1.8) +
  
  # Forward TT controls
  geom_line(
    data = ForwardControl,
    aes(
      x = n,
      y = mean_r2,
      colour = transform,
      group = transform
    ),
    inherit.aes = FALSE,
    linetype = "dotted",
    linewidth = 0.7
  ) +
  
  geom_linerange(
    data = ForwardControl,
    aes(
      x = n,
      ymin = lower,
      ymax = upper,
      colour = transform
    ),
    inherit.aes = FALSE,
    linewidth = 0.4,
    alpha = 0.6
  ) +
  
  # Inverse BB control
  geom_line(
    data = InverseControl,
    aes(
      x = n,
      y = mean_r2
    ),
    inherit.aes = FALSE,
    colour = "black",
    linetype = "dashed",
    linewidth = 0.7
  ) +
  
  geom_linerange(
    data = InverseControl,
    aes(
      x = n,
      ymin = lower,
      ymax = upper
    ),
    inherit.aes = FALSE,
    colour = "black",
    linewidth = 0.4,
    alpha = 0.65
  ) +
  
  facet_grid(
    direction ~ nonlinearity_panel
  ) +
  
  scale_x_log10(
    breaks = sort(unique(LearningCurvePlot$n)),
    labels = sort(unique(LearningCurvePlot$n))
  ) +
  
  labs(
    x = "Sample size",
    y = expression("Test " * R^2),
    colour = NULL,
    shape = NULL
  ) +
  
  theme_classic(base_size = 9.5) +
  
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 9.5
    ),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p_learning_ci

ggsave(
  "figures/nonlinearity_r2_n_CI.pdf",
  p_learning_ci,
  width = 5.5,
  height = 4.4,
  units = "in",
  device = cairo_pdf
)

###XGBoost
XGBMatchedPlot = Matchedraw %>%
  filter(model == "xgb") %>%
  left_join(
    StrengthLookup,
    by = c("transform", "strength")
  )

XGBForwardBT = XGBMatchedPlot %>%
  filter(
    observed_representation == "base",
    signal_representation == "transformed",
    near(nonlinearity, 0.05) |
      near(nonlinearity, 0.10) |
      near(nonlinearity, 0.15) |
      near(nonlinearity, 0.20) |
      near(nonlinearity, 0.30)
  ) %>%
  select(
    rep, n, transform, strength, nonlinearity,
    r2_BT = r2_test
  )


XGBForwardTT = XGBMatchedPlot %>%
  filter(
    observed_representation == "transformed",
    signal_representation == "transformed"
  ) %>%
  select(
    rep, n, transform, strength,
    r2_TT = r2_test
  )


XGBForwardDelta = XGBForwardBT %>%
  left_join(
    XGBForwardTT,
    by = c("rep", "n", "transform", "strength")
  ) %>%
  mutate(
    delta_r2 = r2_TT - r2_BT,
    direction = "Forward map"
  )

XGBInverseTB = XGBMatchedPlot %>%
  filter(
    observed_representation == "transformed",
    signal_representation == "base",
    near(nonlinearity, 0.05) |
      near(nonlinearity, 0.10) |
      near(nonlinearity, 0.15) |
      near(nonlinearity, 0.20) |
      near(nonlinearity, 0.30)
  ) %>%
  select(
    rep, n, transform, strength, nonlinearity,
    r2_TB = r2_test
  )



XGBInverseBB = XGBMatchedPlot %>%
  filter(
    observed_representation == "base",
    signal_representation == "base"
  ) %>%
  group_by(rep, n) %>%
  summarize(
    r2_BB = mean(r2_test, na.rm = TRUE),
    .groups = "drop"
  )


XGBInverseDelta = XGBInverseTB %>%
  left_join(
    XGBInverseBB,
    by = c("rep", "n")
  ) %>%
  mutate(
    delta_r2 = r2_BB - r2_TB,
    direction = "Inverse map"
  )

XGBDeltaPlot = bind_rows(
  XGBForwardDelta,
  XGBInverseDelta
) %>%
  group_by(
    n, transform, nonlinearity, direction
  ) %>%
  summarize(
    mean_delta_r2 = mean(delta_r2, na.rm = TRUE),
    se_delta_r2 = sd(delta_r2, na.rm = TRUE) / sqrt(sum(!is.na(delta_r2))),
    .groups = "drop"
  ) %>%
  mutate(
    nonlinearity_panel = factor(
      nonlinearity,
      levels = c(0.05, 0.10, 0.15, 0.20, 0.30),
      labels = c(
        "Nonlinearity = 0.05",
        "Nonlinearity = 0.10",
        "Nonlinearity = 0.15",
        "Nonlinearity = 0.20",
        "Nonlinearity = 0.30"
      )
    )
  )

p_xgb_delta = ggplot(
  XGBDeltaPlot,
  aes(
    x = n,
    y = mean_delta_r2,
    colour = transform,
    shape = transform,
    linetype = direction,
    group = interaction(transform, direction)
  )
) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.5
  ) +
  
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  
  facet_wrap(
    ~ nonlinearity_panel,
    ncol = 3
  ) +
  
  scale_x_log10(
    breaks = sort(unique(XGBDeltaPlot$n)),
    labels = sort(unique(XGBDeltaPlot$n))
  ) +
  
  labs(
    x = "Sample size",
    y = expression(Delta * R^2),
    colour = NULL,
    shape = NULL,
    linetype = NULL
  ) +
  
  theme_classic(base_size = 9.5) +
  
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 9.5
    ),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5),
    axis.text.x = element_text(angle = 45, hjust = 1),
  )

p_xgb_delta

ggsave(
  "figures/xgb_delta_r2_n.pdf",
  p_xgb_delta,
  width = 5.5,
  height = 4.4,
  units = "in",
  device = cairo_pdf
)

##Appendix Plot
LearningCurvePlotAppendix = MatchedPlot %>%
  filter(
    model == "ann",
    near(nonlinearity, 0.05) | near(nonlinearity, 0.15) | near(nonlinearity, 0.2),
    (observed_representation == "base" &
       signal_representation == "transformed") |
      (observed_representation == "transformed" &
         signal_representation == "base")
  ) %>%
  mutate(
    direction = case_when(
      observed_representation == "base" &
        signal_representation == "transformed" ~ "Forward map",
      
      observed_representation == "transformed" &
        signal_representation == "base" ~ "Inverse map"
    ),
    
    nonlinearity_panel = case_when(
      near(nonlinearity, 0.05) ~ "Nonlinearity = 0.05",
      near(nonlinearity, 0.15) ~ "Nonlinearity = 0.15",
      near(nonlinearity, 0.2) ~ "Nonlinearity = 0.2"
    )
  )

ForwardControlAppendix = MatchedPlot %>%
  filter(
    model == "ann",
    near(nonlinearity, 0.05) | near(nonlinearity, 0.15) | near(nonlinearity, 0.2),,
    observed_representation == "transformed",
    signal_representation == "transformed"
  ) %>%
  mutate(
    direction = "Forward map",
    
    nonlinearity_panel = case_when(
      near(nonlinearity, 0.05) ~ "Nonlinearity = 0.05",
      near(nonlinearity, 0.15) ~ "Nonlinearity = 0.15",
      near(nonlinearity, 0.2) ~ "Nonlinearity = 0.2"
    )
  )

InverseControlAppendix = MatchedPlot %>%
  filter(
    model == "ann",
    observed_representation == "base",
    signal_representation == "base"
  ) %>%
  group_by(n) %>%
  summarize(
    mean_r2 = mean(mean_r2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  crossing(
    direction = "Inverse map",
    nonlinearity_panel = c(
      "Nonlinearity = 0.05",
      "Nonlinearity = 0.15",
      "Nonlinearity = 0.2"
    )
  )


p_learningAppendix = ggplot(
  LearningCurvePlotAppendix,
  aes(
    x = n,
    y = mean_r2,
    colour = transform,
    shape = transform,
    group = transform
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  
  # Forward-map TT matched controls
  geom_line(
    data = ForwardControlAppendix,
    aes(
      x = n,
      y = mean_r2,
      colour = transform,
      group = transform
    ),
    inherit.aes = FALSE,
    linetype = "dotted",
    linewidth = 0.7
  ) +
  
  # Inverse-map BB matched control
  geom_line(
    data = InverseControlAppendix,
    aes(
      x = n,
      y = mean_r2
    ),
    inherit.aes = FALSE,
    colour = "black",
    linetype = "dashed",
    linewidth = 0.7
  ) +
  
  facet_grid(
    direction ~ nonlinearity_panel
  ) +
  
  scale_x_log10(
    breaks = c(250, 500, 1000, 2000, 5000, 10000),
    labels = scales::label_comma()
  ) +
  
  labs(
    x = "Sample size",
    y = expression("Test " * R^2),
    colour = NULL,
    shape = NULL
  ) +
  
  theme_classic(base_size = 9.5) +
  
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 9.5
    ),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5),
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )
p_learningAppendix

ggsave(
  "figures/nonlinearity_r2_n_appendix.pdf",
  p_learningAppendix,
  width = 5.5,
  height = 4.4,
  units = "in",
  device = cairo_pdf
)


##Spectral energy Allocation for inverse/forward
Spectral = read.csv("results/matched_hermite_metrics.csv")
Inv_Spectral = read.csv("results/inverse_polynomial_metrics.csv")


ForwardEnergy = hermite %>%
  group_by(transform)%>%
  mutate(
    higher_energy = 1 - energy1 - energy2 - energy3,
    higher_energy_share = higher_energy/(1-energy1)
  ) %>%
  select(
    transform,
    strength,
    forward_energy = higher_energy,
    higher_energy_share
  ) %>%
  summarize(
    mean_higher_energy_share=mean(higher_energy_share)
  )

InverseEnergy = inverse %>%
  group_by(transform)%>%
  mutate(
    higher_energy = 1 - inv_energy1 - inv_energy2 - inv_energy3,
    higher_energy_share = higher_energy/(1-inv_energy1)
  ) %>%
  select(
    transform,
    strength,
    inverse_energy = higher_energy,
    higher_energy_share
  )%>%
  summarize(
    mean_higher_energy_share=mean(higher_energy_share)
  )

FullForwardEnergy = hermite %>%
  mutate(
    higher_energy = 1 - energy1 - energy2 - energy3,
    higher_energy_share = higher_energy/(1-energy1)
  ) %>%
  select(
    transform,
    strength,
    forward_energy = higher_energy,
    higher_energy_share
  ) 

FullInverseEnergy = inverse %>%
  mutate(
    higher_energy = 1 - inv_energy1 - inv_energy2 - inv_energy3,
    higher_energy_share = higher_energy/(1-inv_energy1)
  ) %>%
  select(
    transform,
    strength,
    inverse_energy = higher_energy,
    higher_energy_share
  )

##ResNet
ResNet = read_excel("results/matched_nonlinearity_resnet_n500_p50_a10_fixedV2.xlsx")



ResNetPlot = ResNet %>%
  group_by(transform, strength,
           observed_representation, signal_representation) %>%
  summarize(
    mean_r2 = mean(r2_test)
  )%>%
  filter(
    (observed_representation == "base" &
       signal_representation == "transformed") |
      (observed_representation == "transformed" &
         signal_representation == "base")
  ) %>%
  mutate(
    direction = case_when(
      observed_representation == "base" &
        signal_representation == "transformed" ~ "Forward map",
      observed_representation == "transformed" &
        signal_representation == "base" ~ "Inverse map"
    ),
    direction = factor(
      direction,
      levels = c("Forward map", "Inverse map")
    )
  ) %>%
  left_join(StrengthLookup,
                      by=c("transform","strength"))

p_resnet = ggplot(
  ResNetPlot,
  aes(
    x = nonlinearity,
    y = mean_r2,
    colour = transform,
    shape = transform,
    group = transform
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ direction, nrow = 1) +
  labs(
    x = "Matched nonlinearity",
    y = expression("Test " * R^2),
    colour = NULL,
    shape = NULL
  ) +
  theme_classic(base_size = 9.5) +
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 9.5),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5)
  )

p_resnet

ggsave(
  "figures/resnet_nonlinearity_curves.pdf",
  p_resnet,
  width = 5.5,
  height = 4.3,
  units = "in",
  device = cairo_pdf
)

##TabICL

TabICL = read_excel("results/matched_nonlinearity_tabicl_n2000_p50_a10.xlsx")

TabICLPlot = TabICL %>%
  group_by(transform, strength,
           observed_representation, signal_representation) %>%
  summarize(
    mean_r2 = mean(r2_test)
  )%>%
  filter(
    (observed_representation == "base" &
       signal_representation == "transformed") |
      (observed_representation == "transformed" &
         signal_representation == "base")
  ) %>%
  mutate(
    direction = case_when(
      observed_representation == "base" &
        signal_representation == "transformed" ~ "Forward map",
      observed_representation == "transformed" &
        signal_representation == "base" ~ "Inverse map"
    ),
    direction = factor(
      direction,
      levels = c("Forward map", "Inverse map")
    )
  ) %>%
  left_join(StrengthLookup,
            by=c("transform","strength"))

p_TabICL = ggplot(
  TabICLPlot,
  aes(
    x = nonlinearity,
    y = mean_r2,
    colour = transform,
    shape = transform,
    group = transform
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ direction, nrow = 1) +
  labs(
    x = "Matched nonlinearity",
    y = expression("Test " * R^2),
    colour = NULL,
    shape = NULL
  ) +
  theme_classic(base_size = 9.5) +
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 9.5),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5)
  )

p_TabICL

ggsave(
  "figures/tabicl_nonlinearity_curves.pdf",
  p_TabICL,
  width = 5.5,
  height = 4.3,
  units = "in",
  device = cairo_pdf
)


##Population Control
Population = read_excel("results/population_normalized_matched_nonlinearity_transform_mismatch_n500_p50_a10.xlsx")

PopPlot = Population %>%
  filter(model == "ann") %>%
  group_by(transform, strength,
           observed_representation, signal_representation, model) %>%
  summarize(
    mean_r2 = mean(r2_test)
  )%>%
  filter(
    (observed_representation == "base" &
       signal_representation == "transformed") |
      (observed_representation == "transformed" &
         signal_representation == "base")
  ) %>%
  mutate(
    direction = case_when(
      observed_representation == "base" &
        signal_representation == "transformed" ~ "Forward map",
      observed_representation == "transformed" &
        signal_representation == "base" ~ "Inverse map"
    ),
    direction = factor(
      direction,
      levels = c("Forward map", "Inverse map")
    )
  ) %>%
  left_join(StrengthLookup,
            by=c("transform","strength"))

p_pop = ggplot(
  PopPlot,
  aes(
    x = nonlinearity,
    y = mean_r2,
    colour = transform,
    shape = transform,
    group = transform
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ direction, nrow = 1) +
  labs(
    x = "Matched nonlinearity",
    y = expression("Test " * R^2),
    colour = NULL,
    shape = NULL
  ) +
  theme_classic(base_size = 9.5) +
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 9.5),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5)
  )

p_pop

ggsave(
  "figures/population_curves.pdf",
  p_pop,
  width = 5.5,
  height = 2.8,
  units = "in",
  device = cairo_pdf
)

##Optimization plot
OptimizationRaw = read_excel(
  "results/ann_optimization_diagnostics_modern_mismatch.xlsx",
  sheet = "raw_results"
)

OptimizationPaired = OptimizationRaw %>%
  filter(
    stage == "optimization_budget",
    architecture_name == "baseline_64_32",
    status == "ok",
    nonlinearity_level %in% c(0.2, 0.3)
  ) %>%
  select(
    rep,
    transform,
    nonlinearity_level,
    max_iter,
    condition,
    r2_test
  ) %>%
  pivot_wider(
    names_from = condition,
    values_from = r2_test
  ) %>%
  mutate(
    forward_delta = TT - BT,
    inverse_delta = BB - TB
  ) %>%
  pivot_longer(
    cols = c(forward_delta, inverse_delta),
    names_to = "direction",
    values_to = "delta_r2"
  ) %>%
  mutate(
    direction = case_when(
      direction == "forward_delta" ~ "Forward map",
      direction == "inverse_delta" ~ "Inverse map"
    ),
    
    nonlinearity_panel = case_when(
      near(nonlinearity_level, 0.2) ~ "Nonlinearity = 0.20",
      near(nonlinearity_level, 0.3) ~ "Nonlinearity = 0.30"
    )
  )

OptimizationPlot = OptimizationPaired %>%
  group_by(
    max_iter,
    transform,
    nonlinearity_panel,
    direction
  ) %>%
  summarize(
    mean_delta_r2 = mean(delta_r2, na.rm = TRUE),
    se_delta_r2 = sd(delta_r2, na.rm = TRUE) /
      sqrt(sum(!is.na(delta_r2))),
    .groups = "drop"
  )
OptimizationPlot = OptimizationPlot %>%
  mutate(
    max_iter_factor = factor(
      max_iter,
      levels = c(250, 500, 1000, 2000)
    )
  )

pd = position_dodge(width = 0.55)

p_optimization = ggplot(
  OptimizationPlot,
  aes(
    x = max_iter_factor,
    y = mean_delta_r2,
    colour = transform,
    shape = transform,
    group = transform
  )
) +
  geom_linerange(
    aes(
      ymin = mean_delta_r2 - 1.96 * se_delta_r2,
      ymax = mean_delta_r2 + 1.96 * se_delta_r2
    ),
    position = pd,
    linewidth = 0.4,
    alpha = 0.55
  ) +
  
  geom_point(
    position = pd,
    size = 2
  ) +
  
  facet_grid(
    direction ~ nonlinearity_panel
  ) +
  
  labs(
    x = "Maximum training iterations",
    y = expression(Delta * R^2),
    colour = NULL,
    shape = NULL
  ) +
  
  theme_classic(base_size = 9.5) +
  
  theme(
    legend.position = "top",
    strip.background = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 9.5
    ),
    legend.text = element_text(size = 8.5),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5)
  )

p_optimization

ggsave(
  "figures/optimization_budget_delta_r2.pdf",
  p_optimization,
  width = 5.5,
  height = 3.6,
  units = "in",
  device = cairo_pdf
)

##Quantile Transformation Experiment
Raw2000 <- read_excel(
  "results/matched_nonlinearity_transform_mismatch_n2000_p50_a10.xlsx",
  sheet = "results"
)

Raw10000 <- read_excel(
  "results/matched_nonlinearity_transform_mismatch_n10000_p50_a10.xlsx",
  sheet = "results"
)

RawResults <- bind_rows(Raw2000, Raw10000) %>%
  filter(
    rep <= 10,
    status == "ok",
    model == "ann"
  )

Quantile2000 <- read_excel(
  "results/matched_nonlinearity_transform_mismatch_quantile_n2000_p50_a10.xlsx",
  sheet = "results"
)

Quantile10000 <- read_excel(
  "results/matched_nonlinearity_transform_mismatch_quantile_n10000_p50_a10.xlsx",
  sheet = "results"
)

QuantileResults <- bind_rows(Quantile2000, Quantile10000) %>%
  filter(
    rep <= 10,
    status == "ok",
    model %in% c("ann_qnormal", "ann_quniform")
  )

add_condition <- function(df) {
  df %>%
    mutate(
      condition = case_when(
        observed_representation == "base" &
          signal_representation == "base" ~ "BB",
        
        observed_representation == "base" &
          signal_representation == "transformed" ~ "BT",
        
        observed_representation == "transformed" &
          signal_representation == "base" ~ "TB",
        
        observed_representation == "transformed" &
          signal_representation == "transformed" ~ "TT"
      )
    )
}

RawResults <- add_condition(RawResults)
QuantileResults <- add_condition(QuantileResults)

StrengthLookup <- bind_rows(
  tibble(
    transform = "cubic",
    strength = sort(unique(RawResults$strength[RawResults$transform == "cubic"])),
    eta = c(0.05, 0.10, 0.15, 0.20, 0.30)
  ),
  tibble(
    transform = "quintic",
    strength = sort(unique(RawResults$strength[RawResults$transform == "quintic"])),
    eta = c(0.05, 0.10, 0.15, 0.20, 0.30)
  ),
  tibble(
    transform = "exp",
    strength = sort(unique(RawResults$strength[RawResults$transform == "exp"])),
    eta = c(0.05, 0.10, 0.15, 0.20, 0.30)
  ),
  tibble(
    transform = "quad",
    strength = sort(unique(RawResults$strength[RawResults$transform == "quad"])),
    eta = c(0.05, 0.10, 0.15, 0.20, 0.30)
  )
)

RawResults <- RawResults %>%
  left_join(StrengthLookup, by = c("transform", "strength"))

QuantileResults <- QuantileResults %>%
  left_join(StrengthLookup, by = c("transform", "strength")) %>%
  mutate(
    preprocessing = recode(
      model,
      ann_qnormal = "Quantile -> Normal",
      ann_quniform = "Quantile -> Uniform"
    )
  )

ConditionEffect <- QuantileResults %>%
  select(
    n, rep, transform, strength, eta,
    preprocessing, condition,
    r2_quantile = r2_test
  ) %>%
  left_join(
    RawResults %>%
      select(
        n, rep, transform, strength, eta,
        condition,
        r2_raw = r2_test
      ),
    by = c("n", "rep", "transform", "strength", "eta", "condition")
  ) %>%
  mutate(
    delta_r2 = r2_quantile - r2_raw
  )
ConditionTable <- ConditionEffect %>%
  group_by(
    n, eta, preprocessing, condition
  ) %>%
  summarize(
    mean_delta = mean(delta_r2, na.rm = TRUE),
    se_delta = sd(delta_r2, na.rm = TRUE) /
      sqrt(sum(!is.na(delta_r2))),
    .groups = "drop"
  ) %>%
  mutate(
    entry = sprintf("%.3f (%.3f)", mean_delta, se_delta),
    sample_size = paste0("n = ", n),
    eta_label = sprintf("eta = %.2f", eta)
  ) %>%
  select(sample_size, eta_label, preprocessing, condition, entry) %>%
  pivot_wider(
    names_from = condition,
    values_from = entry
  ) %>%
  arrange(sample_size, eta_label, preprocessing)

ConditionTable

ConditionHeatmap <- ConditionEffect %>%
  group_by(
    n, eta, preprocessing, condition
  ) %>%
  summarize(
    mean_delta = mean(delta_r2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    sample_size = factor(
      paste0("n = ", n),
      levels = c("n = 2000", "n = 10000")
    ),
    eta_label = factor(
      sprintf("%.2f", eta),
      levels = sprintf("%.2f", c(0.05, 0.10, 0.15, 0.20, 0.30))
    ),
    preprocessing = factor(
      preprocessing,
      levels = c("Quantile -> Normal", "Quantile -> Uniform")
    ),
    condition = factor(
      condition,
      levels = c("BB", "BT", "TB", "TT")
    ),
    label = sprintf("%.3f", mean_delta)
  )

p_condition_heatmap <- ggplot(
  ConditionHeatmap,
  aes(
    x = condition,
    y = eta_label,
    fill = mean_delta
  )
) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(
    aes(label = label),
    size = 2.8
  ) +
  facet_grid(
    preprocessing ~ sample_size
  ) +
  scale_fill_gradient2(
    low = "#B2182B",
    mid = "white",
    high = "#2166AC",
    midpoint = 0
  ) +
  labs(
    x = "Condition",
    y = expression("Matched nonlinearity " * eta),
    fill = expression(Delta * R^2)
  ) +
  theme_classic(base_size = 9.5) +
  theme(
    legend.position = "right",
    strip.background = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 9.5
    ),
    axis.title = element_text(size = 9.5),
    axis.text = element_text(size = 8.5)
  )

p_condition_heatmap

ggsave(
  "figures/quantile_condition_heatmap.pdf",
  p_condition_heatmap,
  width = 5.5,
  height = 3.6,
  units = "in",
  device = cairo_pdf
)
