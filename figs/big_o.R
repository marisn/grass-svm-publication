library(ggplot2)
library(minpack.lm)
library(AICcmodavg)

timingrg <- read.csv("timing.csv", header=T)
timingrg$labels <- paste(timingrg$module, timingrg$tp)

fit_lm <- function(dt) {
    fits <- list(
        lm(rt~I(log(log(cells))), data=dt),
        lm(rt~I(log(cells)), data=dt),
        lm(rt~I(log(cells)^2), data=dt),
        lm(rt~I(cells^0.5), data=dt),
        lm(rt~I(cells * log(cells)), data=dt),
        lm(rt~I(cells^2), data=dt),
        lm(rt~I(cells^2.5), data=dt),
        lm(rt~I(cells^3), data=dt),
        lm(rt~I(cells^3.5), data=dt),
        lm(rt~I(cells^4), data=dt)
    )
    fit.names <- c("log(log(n))", "log(n)", "log(n)^2", "n^0.5", "n*log(n)", "n^2", "n^2.5", "n^3", "n^3.5", "n^4")
    aic <- aictab(cand.set = fits, modnames = fit.names)
    print(aic)
    return(aic)
}

fit <- fit_lm

print(paste("ML2 train:",  fit(timingrg[timingrg$labels == "ML2 t",])$Modnames[1]))
print(paste("SVMn train:", fit(timingrg[timingrg$labels == "SVMn t",])$Modnames[1]))
print(paste("SVMh train:", fit(timingrg[timingrg$labels == "SVMh t",])$Modnames[1]))
print(paste("SMAP train:", fit(timingrg[timingrg$labels == "SMAP t",])$Modnames[1]))
print(paste("MLC train:",  fit(timingrg[timingrg$labels == "MLC t",])$Modnames[1]))
print(paste("ML2 pred:",   fit(timingrg[timingrg$labels == "ML2 p",])$Modnames[1]))
print(paste("SVM pred:",   fit(timingrg[timingrg$labels == "SVM p",])$Modnames[1]))
print(paste("SMAP pred:",  fit(timingrg[timingrg$labels == "SMAP p",])$Modnames[1]))
print(paste("MLC pred:",   fit(timingrg[timingrg$labels == "MLC p",])$Modnames[1]))

fig5 <- ggplot(data=timingrg, aes(x=cells, y=rt, color=labels, linetype=tp)) + geom_point() +
    scale_x_continuous(limit=c(0.001, 45000)) +
    scale_y_continuous(limit=c(0.001, 15)) +
    scale_color_brewer(palette="Set1") +
    geom_line(aes(x=cells, y=rt, group=labels, linetype=tp),
        data=timingrg[timingrg$labels == "MLC t" | timingrg$labels == "SMAP p" | timingrg$labels == "MLC p",],
        stat="smooth", method="lm",
        formula=y~I(x^0.5), fullrange=TRUE, se=FALSE, size=1.0, alpha=0.5) +
    geom_line(aes(x=cells, y=rt, group=labels, linetype=tp),
        data=timingrg[timingrg$labels == "ML2 t" | timingrg$labels == "ML2 p",],
        stat="smooth", method="lm",
        formula=y~I(x^2), fullrange=TRUE, se=FALSE, size=1.0, alpha=0.5) +
    geom_line(aes(x=cells, y=rt, group=labels, linetype=tp),
        data=timingrg[timingrg$labels == "SVMn t",],
        stat="smooth", method="lm",
        formula=y~I(x^2.5), fullrange=TRUE, se=FALSE, size=1.0, alpha=0.5) +
    geom_line(aes(x=cells, y=rt, group=labels, linetype=tp),
        data=timingrg[timingrg$labels == "SVMh t",],
        stat="smooth", method="lm",
        formula=y~I(x^3), fullrange=TRUE, se=FALSE, size=1.0, alpha=0.5) +
    geom_line(aes(x=cells, y=rt, group=labels, linetype=tp),
        data=timingrg[timingrg$labels == "SMAP t" | timingrg$labels == "SVM p",],
        stat="smooth", method="lm",
        formula=y~I(x*log(x)), fullrange=TRUE, se=FALSE, size=1.0, alpha=0.5) +
    theme(legend.position="bottom", legend.title=element_blank(), legend.box="vertical", legend.margin=margin()) +
    xlab("Pixel count") + ylab("Wall time (s)") +
    guides(color=guide_legend(nrow=3, byrow=TRUE))

ggsave("fig_5.png", width=80, height=105, units="mm", dpi=600, fig5)
