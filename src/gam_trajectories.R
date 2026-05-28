#!/usr/bin/env Rscript
# gam_trajectories.R — v2.2：对 43 个体测/生化拟合 mgcv::gam(feature ~ s(age), family=scat())
#
# 方案 §12：scat (scaled-t) 重尾族对异常值稳健；s(age, bs="tp", k=20) 薄板样条
# 输入：parquet（含 age 列 + N 个 feature 列）
# 输出：
#   gam_summary_{sex}.csv   每特征一行：edf, Ref.df, F, p_value, p_bh, sig_flag
#   gam_curves_{sex}.png    43 个 facet 小多图（年龄网格 predict 曲线 + 95% CI）
#   gam_heatmap_{sex}.png   z-score 后的轨迹热图（层次聚类）
#
# 由 src/run_gam.py 通过 subprocess 调用，CLI 顺序参数：
#   Rscript gam_trajectories.R <input_parquet> <sex_label> <out_dir> \
#                              <age_col> <k> <fdr_alpha> <edf_thresh> \
#                              <age_min> <age_max> <age_n>

suppressPackageStartupMessages({
  library(mgcv)
  library(arrow)
  library(data.table)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 10) {
  stop("用法: Rscript gam_trajectories.R <input.parquet> <sex_label> <out_dir> ",
       "<age_col> <k> <fdr_alpha> <edf_thresh> <age_min> <age_max> <age_n>")
}
input_path  <- args[1]
sex_label   <- args[2]
out_dir     <- args[3]
age_col     <- args[4]
k_splines   <- as.integer(args[5])
fdr_alpha   <- as.numeric(args[6])
edf_thresh  <- as.numeric(args[7])
age_min     <- as.numeric(args[8])
age_max     <- as.numeric(args[9])
age_n       <- as.integer(args[10])

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[gam] sex=%s | input=%s | k=%d | age_grid=[%g,%g,n=%d]\n",
            sex_label, input_path, k_splines, age_min, age_max, age_n))

dat <- arrow::read_parquet(input_path)
setDT(dat)
stopifnot(age_col %in% names(dat))
feat_cols <- setdiff(names(dat), age_col)
n_feat <- length(feat_cols)
cat(sprintf("[gam] n_rows=%d  n_features=%d\n", nrow(dat), n_feat))

age_grid <- seq(age_min, age_max, length.out = age_n)
grid_df <- data.frame(age = age_grid)
names(grid_df) <- age_col

# —— fit GAMs ——
summary_rows <- vector("list", n_feat)
curve_rows   <- vector("list", n_feat)
t0 <- Sys.time()
for (i in seq_along(feat_cols)) {
  fcol <- feat_cols[i]
  sub <- dat[!is.na(get(fcol)) & !is.na(get(age_col)),
             .(y = get(fcol), age = get(age_col))]
  setnames(sub, "age", age_col)
  # 模型公式：y ~ s(age, bs='tp', k=k_splines)
  form <- as.formula(sprintf("y ~ s(%s, bs='tp', k=%d)", age_col, k_splines))
  fit <- tryCatch(
    gam(form, data = sub, family = scat(), method = "REML"),
    error = function(e) {
      cat(sprintf("  [warn] %s scat() failed (%s); falling back to gaussian\n",
                  fcol, conditionMessage(e)))
      gam(form, data = sub, family = gaussian(), method = "REML")
    }
  )
  smr <- summary(fit)
  s_tab <- smr$s.table   # rows: smooth terms; cols 依族不同：gaussian 有 F，scat 等有 Chi.sq
  cnames <- colnames(s_tab)
  stat_col <- intersect(c("F", "Chi.sq"), cnames)[1]
  summary_rows[[i]] <- data.table(
    feature = fcol,
    n       = nrow(sub),
    edf     = unname(s_tab[1, "edf"]),
    ref_df  = unname(s_tab[1, "Ref.df"]),
    stat_name = stat_col,
    stat_value = unname(s_tab[1, stat_col]),
    p_value = unname(s_tab[1, "p-value"]),
    deviance_explained = smr$dev.expl,
    family  = fit$family$family
  )
  # —— 预测网格 ——
  pred <- predict(fit, newdata = grid_df, se.fit = TRUE)
  curve_rows[[i]] <- data.table(
    feature = fcol,
    age     = age_grid,
    fit     = pred$fit,
    se      = pred$se.fit
  )
  if (i %% 5 == 0 || i == n_feat) {
    cat(sprintf("  fit %d/%d (%s)  elapsed=%.1fs\n",
                i, n_feat, fcol,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
}

summary_dt <- rbindlist(summary_rows)
curves_dt  <- rbindlist(curve_rows)

# —— BH-FDR ——
summary_dt[, p_bh := p.adjust(p_value, method = "BH")]
summary_dt[, sig_flag := (p_bh < fdr_alpha) & (edf > edf_thresh)]
setorder(summary_dt, p_bh)

summary_path <- file.path(out_dir, sprintf("gam_summary_%s.csv", sex_label))
fwrite(summary_dt, summary_path)
cat(sprintf("[gam] -> %s  (%d/%d significant)\n",
            summary_path,
            sum(summary_dt$sig_flag, na.rm = TRUE),
            nrow(summary_dt)))

# —— curves 图 (43 facet) ——
curves_dt[, lo := fit - 1.96 * se]
curves_dt[, hi := fit + 1.96 * se]
# 按 feature 名排序，保持图稳定
curves_dt[, feature := factor(feature, levels = feat_cols)]
curves_path <- file.path(out_dir, sprintf("gam_curves_%s.png", sex_label))
n_col_facet <- 6
n_row_facet <- ceiling(n_feat / n_col_facet)
p_curves <- ggplot(curves_dt, aes(x = age, y = fit)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = "steelblue", alpha = 0.3) +
  geom_line(color = "steelblue", linewidth = 0.6) +
  facet_wrap(~ feature, ncol = n_col_facet, scales = "free_y") +
  labs(title = sprintf("GAM trajectories — %s (mgcv scat, k=%d)",
                       sex_label, k_splines),
       x = "Chronological age", y = NULL) +
  theme_minimal(base_size = 8) +
  theme(strip.text = element_text(size = 7),
        panel.grid.minor = element_blank())
ggsave(curves_path, p_curves,
       width = 2.0 * n_col_facet, height = 1.3 * n_row_facet,
       dpi = 130, limitsize = FALSE)
cat(sprintf("[gam] -> %s\n", curves_path))

# —— heatmap (z-score 后的轨迹，按相似性聚类) ——
# 把每个 feature 的 fit 在 age 网格上 z-score，然后按 feature 维度聚类
mat <- dcast(curves_dt, feature ~ age, value.var = "fit")
feat_vec <- mat$feature
mat[, feature := NULL]
mat_arr <- as.matrix(mat)
rownames(mat_arr) <- as.character(feat_vec)
# z-score along each row (across age)
mat_z <- t(scale(t(mat_arr)))
# 层次聚类
ord <- hclust(dist(mat_z, method = "euclidean"), method = "ward.D2")$order
mat_z_ord <- mat_z[ord, , drop = FALSE]

heat_dt <- as.data.table(mat_z_ord, keep.rownames = "feature")
heat_long <- melt(heat_dt, id.vars = "feature",
                  variable.name = "age", value.name = "z")
heat_long[, age := as.numeric(as.character(age))]
heat_long[, feature := factor(feature, levels = rownames(mat_z_ord))]

heatmap_path <- file.path(out_dir, sprintf("gam_heatmap_%s.png", sex_label))
p_heat <- ggplot(heat_long, aes(x = age, y = feature, fill = z)) +
  geom_tile() +
  scale_fill_gradient2(low = "navy", mid = "white", high = "firebrick",
                       midpoint = 0, name = "z(GAM fit)") +
  labs(title = sprintf("GAM trajectory heatmap — %s (Ward clustering)",
                       sex_label),
       x = "Chronological age", y = NULL) +
  theme_minimal(base_size = 9) +
  theme(panel.grid = element_blank())
ggsave(heatmap_path, p_heat,
       width = 8, height = 0.18 * n_feat + 2, dpi = 130, limitsize = FALSE)
cat(sprintf("[gam] -> %s\n", heatmap_path))

cat(sprintf("[gam] done | total elapsed=%.1fs\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))
