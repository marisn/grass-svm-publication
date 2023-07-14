library(akima)
library(ggplot2)
library(metR)
library(gridExtra)

svm_default <- data.frame(C=1, gamma=1)
svm <- read.csv("training_svm_p1.csv", header=T)
lsvm <- data.frame(log(svm$C), log(svm$gamma), svm$mcc, svm$oac)
names(lsvm) <- c("C", "gamma", "mcc", "oac")
fld <- with(lsvm, interp(x=C, y=gamma, z=oac, duplicate="mean", nx=300, ny=300))
grdf <- data.frame(x = rep(fld$x, nrow(fld$z)), y = rep(fld$y, each = ncol(fld$z)), z = as.numeric(fld$z))
v <- ggplot(grdf, aes(x, y, z=z)) + geom_contour_filled(binwidth=1, ) +
    metR::scale_fill_discretised(name="%", low="#7E1F08", high="#FFF7B8",
        breaks=c(66, 68, 70, 72, 74, 76, 78, 80, 82, 84)) +
    theme(legend.position="bottom", legend.key.width=unit(30,"pt")) +
    geom_contour(binwidth=1, color="grey30") +
    xlab("ln(C)") + ylab("ln(gamma)") +
    geom_point(data=svm_default, aes(x=C, y=gamma, z=NULL), color="black", size=3, fill="grey80", shape=23)

smap_default <- data.frame(nsigs=c(5), blocksize=c(1024))
smap <- read.csv("train_smap_final.csv", header=T)
names(smap) <- c("nsigs", "blocksize", "mcc", "kappa", "oac")
lsmap <- data.frame(smap$nsigs, log(smap$blocksize), smap$mcc, smap$oac)
names(lsmap) <- c("nsigs", "blocksize", "mcc", "oac")
sfld <- with(smap[smap$oac > 49,], interp(x=nsigs, y=blocksize, z=oac, duplicate="mean", nx=300, ny=300))
sgrdf <- data.frame(x = rep(sfld$x, nrow(sfld$z)), y = rep(sfld$y, each = ncol(sfld$z)), z = as.numeric(sfld$z))
s <- ggplot() + geom_contour_filled(data=sgrdf, aes(x, y, z=z), binwidth=2) +
    metR::scale_fill_discretised(name="%", low="#7E1F08", high="#FFF7B8") +
    theme(legend.position="bottom", legend.key.width=unit(30,"pt")) +
    geom_contour(data=sgrdf, aes(x, y, z=z), binwidth=2, color="grey30") +
    xlab("maxsig") + ylab("blocksize") +
    geom_point(data=smap[smap$oac <= 49,], aes(x=nsigs, y=blocksize), color="black", size=2, fill="white", shape=21) +
    geom_point(data=smap_default, aes(x=nsigs, y=blocksize), color="black", size=3, fill="grey", shape=23)

fig2 <- grid.arrange(s, v, ncol=2)
ggsave("fig_2.png", width=180, height=100, units="mm", dpi=600, fig2)
