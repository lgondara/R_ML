install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
install.packages("imager")
install.packages("installr")

## Classify images
require(mxnet)
require(imager)
require(installr)

install.ImageMagick(URL="http://www.imagemagick.org/script/binary-releases.php")
##load pre-trained model
##model can be downloaded from "https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/classifyRealImageWithPretrainedModel.Rmd"
model = mx.model.load("Inception/Inception_BN", iteration=39)

##load mean image
mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])

##try a sample image from imager
im=imager:::load.png("C:/Users/ldpc/Documents/R/win-library/3.2/imager/extdata/parrots.png")
plot(im)


##pre-processing
preproc.image <-function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  yy <- floor((shape[1] - short.edge) / 2) + 1
  yend <- yy + short.edge - 1
  xx <- floor((shape[2] - short.edge) / 2) + 1
  xend <- xx + short.edge - 1
  croped <- im[yy:yend, xx:xend,,]
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized)
  dim(arr) = c(224, 224, 3)
  # substract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

normed <- preproc.image(im, mean.img)

##classify the image
prob <- predict(model, X=normed)
dim(prob)

max.idx <- max.col(t(prob))
max.idx

##read name of classes to see which the result corresponds to
synsets <- readLines("Inception/synset.txt")
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))
