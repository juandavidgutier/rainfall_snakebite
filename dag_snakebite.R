library(ggdag)
library(dagitty)
library(lavaan)
library(dplyr)
library(GGally)
library(tidyr)



#implied Conditional Independencies
dataset <- read.csv("D:/clases/UDES/articulo accidente ofidico/uniAndes/ci/manuscript/data_top300.csv")
dataset <- select(dataset, excess, Rain, soi, Esoi, SST3, SST4, SST34, SST12, NATL, SATL, TROP, forest,	Rgdp)
dataset <- dataset[complete.cases(dataset), ] 
str(dataset)

#descriptive analysis
#ggpairs(dataset)

#DAG 
dag <- dagitty('dag {
excess [pos="0, 0.5"]
Rain  [pos="-1, 0.5"]

soi [pos="-1.6, 1.1"]
Esoi [pos="-1.7, 1.2"]
SST3 [pos="-1.8, 1.3"]
SST4 [pos="-1.9, 1.4"]
SST34 [pos="-2, 1.5"]
SST12 [pos="-2.1, 1.6"]
NATL [pos="-2.2, 1.7"]
SATL [pos="-2.3, 1.8"]
TROP [pos="-2.4, 1.9"]

Rgdp [pos="-0.2, -0.4"]
forest [pos="-1.7, -0.5"]

SST12 -> SST3
SST12 -> SST34
SST12 -> SST4
SST12 -> soi
SST12 -> Esoi
SST12 -> NATL
SST12 -> SATL
SST12 -> TROP

SST3 -> SST34
SST3 -> SST4
SST3 -> soi
SST3 -> Esoi
SST3 -> NATL
SST3 -> SATL
SST3 -> TROP

SST34 -> SST4
SST34 -> soi
SST34 -> Esoi
SST34 -> NATL
SST34 -> SATL
SST34 -> TROP

SST4 -> soi
SST4 -> Esoi
SST4 -> NATL
SST4 -> SATL
SST4 -> TROP

soi -> Esoi
soi -> NATL
soi -> SATL
soi -> TROP

Esoi -> NATL
Esoi -> SATL
Esoi -> TROP

NATL -> SATL
NATL -> TROP

SATL -> TROP

SST12 -> Rain
SST3 -> Rain
SST34 -> Rain
SST4 -> Rain
soi -> Rain
Esoi -> Rain
NATL -> Rain
SATL -> Rain
TROP -> Rain
forest -> Rain

SST12 -> excess
SST3 -> excess
SST34 -> excess
SST4 -> excess
soi -> excess
Esoi -> excess
NATL -> excess
SATL -> excess
TROP -> excess
forest -> excess

forest -> Rgdp
Rgdp -> forest

Rain -> Rgdp


Rgdp -> excess

Rain -> excess

}')  


plot(dag)


## check whether any correlations are perfect (i.e., collinearity)
myCov <- cov(dataset)
round(myCov, 2)

myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)

## if not, check for multicollinearity (i.e., is one variable a linear combination of 2+ variables?)
det(myCov) < 0
## or
any(eigen(myCov)$values < 0)


## Independencias condicionales
impliedConditionalIndependencies(dag, max.results=3)
corr <- lavCor(dataset)

summary(corr)

#plot
localTests(dag, sample.cov=corr, sample.nobs=nrow(dataset), max.conditioning.variables=3)
plotLocalTestResults(localTests(dag, sample.cov=corr, sample.nobs=nrow(dataset)), xlim=c(-1,1))


#identification
simple_dag <- dagify(
  excess ~  Rain + SST12 + SST3 + SST34 + SST4 + soi + Eqsoi + NATL + SATL + TROP + forest + Rgdp,
  Rain ~ SST12 + SST3 + SST34 + SST4 + soi + Eqsoi + NATL + SATL +  TROP + forest, 
  Rgdp ~ Rain + forest,
  SST12 ~ SST3 + SST34 + SST4 + soi + Eqsoi + NATL + SATL +  TROP,
  SST3 ~ SST34 + SST4 + soi + Eqsoi + NATL + SATL +  TROP,
  SST34 ~ SST4 + soi + Eqsoi + NATL + SATL +  TROP,
  SST4 ~ soi + Eqsoi + NATL + SATL +  TROP,
  soi ~ Eqsoi + NATL + SATL +  TROP,
  Eqsoi ~ NATL + SATL +  TROP,
  NATL ~ SATL +  TROP,
  SATL ~  TROP,
  exposure = "Rain",
  outcome = "excess",
  coords = list(x = c(Rain=2, forest=1, excess=2, SST12=3, SST3=3.1, SST34=3.2, SST4=3.3, soi=3.4, Eqsoi=3.5, NATL=3.6, SATL=3.7, TROP=3.8,
                      Rgdp=3.5),
                y = c(Rain=2, forest=3, excess=1, SST12=3, SST3=3.1, SST34=3.2, SST4=3.3, soi=3.4, Eqsoi=3.5, NATL=3.6, SATL=3.7, TROP=3.8,
                      Rgdp=1.8))
)


# theme_dag
ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()


#adjust
adjustmentSets(simple_dag,  type = "minimal")

ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()

