library(ggplot2)

perclass <- read.csv("per_class_accuracy.csv", header=T)
perclass$ua <- 100 - perclass$commission
perclass$pa <- 100 - perclass$ommission
perclass$code <- factor(perclass$code)
perclass$method <- factor(perclass$method, levels=c("SVM def.", "SVM opt.", "SMAP def.", "SMAP opt.", "ML"))

testing <- read.csv("testing_cell_distribution.csv", header=T)
testing$code <- factor(testing$code)

fig3 <- ggplot() +
    geom_bar(data=perclass, aes(x=code, y=ua, fill=method), stat="identity", position=position_dodge()) + 
    scale_fill_brewer(palette="Paired", name="Method:") +
    geom_point(data=testing, aes(x=code, y=pcnt), shape="—", size=7) +
    theme(legend.position="bottom") +
    xlab("Crop code") + ylab("User accuracy (%)")

ggsave("fig_3.png", width=180, height=60, units="mm", dpi=600, fig3)
