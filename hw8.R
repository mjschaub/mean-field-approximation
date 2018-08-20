# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
setwd('D:/CS498/HW8 - MEAN-FIELD APPROXIMATION/')

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('train-images.idx3-ubyte')
  #test <<- load_image_file('mnist/t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('train-labels.idx1-ubyte')
  #test$y <<- load_label_file('mnist/t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}
show_digit_2 <- function(arr784, col=gray(12:1/12), ...) {
  image(t(matrix(arr784,nrow=28)[,28:1]), col=col, ...)
}

load_mnist()

twenty_images <- array(0,dim=c(20,28,28))
for(i in 1:20)
{
  twenty_images[i,,] <- matrix(train$x[i,1:784],nrow=28)
  for(x in 1:28)
  {
    for(y in 1:28)
    {
      if(twenty_images[i,x,y] >= 128)
        twenty_images[i,x,y] <- 1
      else
        twenty_images[i,x,y] <- -1
      
    }
  }
}

noise_coords <-read.csv('NoiseCoordinates.csv',header=TRUE)
twenty_images_original <- twenty_images

for(i in seq(from=1, to=40, by=2))
{
  for(x in 2:16)
  {
    curr_row <- noise_coords[i,x]+1
    curr_col <- noise_coords[i+1,x]+1
    if(twenty_images[(i+1)/2,curr_col,curr_row] == 1)
      twenty_images[(i+1)/2,curr_col,curr_row] <- -1
    else
      twenty_images[(i+1)/2,curr_col,curr_row] <- 1
    
  }
}


#show_digit(twenty_images[20,,])
#show_digit(twenty_images_original[11,,])

theta_one <- .8
theta_two <- 2

update_order <- read.csv('UpdateOrderCoordinates.csv',header=TRUE)

initial_params <- read.csv('InitialParametersModel.csv',header=FALSE,sep=',')
initial_params <- as.matrix(initial_params)

iters <- 10
epsilon <- 10^(-10)
E_Q_arr <- array(0,dim=c(20,11))
Q_arr <- array(0,dim=c(20,28,28))

  
for(img in seq(from=1, to=40, by=2))
{
  Q_init <- initial_params
  
  for(iter in 1:iters)
  {
    E_Q_log_Q <- 0
    E_Q_log_PHX <- 0
    for(i in 1:28)
    {
      for(j in 1:28)
      {
        E_Q_log_Q <- E_Q_log_Q + (Q_init[i,j]*log(Q_init[i,j]+epsilon)+(1-Q_init[i,j])*log((1-Q_init[i,j])+epsilon))
        neighbors <- c()
        if(i != 1)
          neighbors <- cbind(neighbors,c(i-1,j))
        if(j != 1)
          neighbors <- cbind(neighbors,c(i,j-1))
        if(i != 28)
          neighbors <- cbind(neighbors,c(i+1,j))
        if(j != 28)
          neighbors <- cbind(neighbors,c(i,j+1))
        neighbor_summation <- 0
        for(n in 1:dim(neighbors)[2])
        {
          n_i <- neighbors[1,n]
          n_j <- neighbors[2,n]
          neighbor_summation <- neighbor_summation + theta_one *(2*Q_init[i,j] - 1)*(2*Q_init[n_i,n_j]-1)
        }
        E_Q_log_PHX <- E_Q_log_PHX + neighbor_summation + theta_two * (2*Q_init[i,j] -1)*twenty_images[(img+1)/2,j,i]
      }
    }
    E_Q <- E_Q_log_Q - E_Q_log_PHX
    E_Q_arr[(img+1)/2,iter] <- E_Q
    
    prev_Q_init <- Q_init
    for(x in 2:785)
    {
      curr_row <- update_order[img,x]+1
      curr_col <- update_order[img+1,x]+1
      #print(c(curr_row,curr_col))
      neighbors <- c()
      if(curr_row != 1)
        neighbors <- cbind(neighbors,c(curr_row-1,curr_col))
      if(curr_col != 1)
        neighbors <- cbind(neighbors,c(curr_row,curr_col-1))
      if(curr_row != 28)
        neighbors <- cbind(neighbors,c(curr_row+1,curr_col))
      if(curr_col != 28)
        neighbors <- cbind(neighbors,c(curr_row,curr_col+1))
      
      
      temp_num <- 0
      for(n in 1:dim(neighbors)[2])
      {
        curr_x <- neighbors[1,n]
        curr_y <- neighbors[2,n]
        temp_num <- temp_num + (theta_one * (2 * Q_init[curr_x, curr_y] - 1))
      }
      temp_num <- temp_num + theta_two * twenty_images[(img+1)/2,curr_col, curr_row]
      top_exp <- exp(temp_num)
      bottom_right_exp <- exp(-temp_num)
      pi_i <- (top_exp / (top_exp+bottom_right_exp))
      Q_init[curr_row,curr_col] <- pi_i
      
      
    }
    #Q_init[curr_row,curr_col] <- pi_i
  }
  E_Q_log_Q <- 0
  E_Q_log_PHX <- 0
  for(i in 1:28)
  {
    for(j in 1:28)
    {
      E_Q_log_Q <- E_Q_log_Q + (Q_init[i,j]*log(Q_init[i,j]+epsilon)+(1-Q_init[i,j])*log((1-Q_init[i,j])+epsilon))
      neighbors <- c()
      if(i != 1)
        neighbors <- cbind(neighbors,c(i-1,j))
      if(j != 1)
        neighbors <- cbind(neighbors,c(i,j-1))
      if(i != 28)
        neighbors <- cbind(neighbors,c(i+1,j))
      if(j != 28)
        neighbors <- cbind(neighbors,c(i,j+1))
      neighbor_summation <- 0
      for(n in 1:dim(neighbors)[2])
      {
        n_i <- neighbors[1,n]
        n_j <- neighbors[2,n]
        neighbor_summation <- neighbor_summation + theta_one *(2*Q_init[i,j] - 1)*(2*Q_init[n_i,n_j]-1)
      }
      E_Q_log_PHX <- E_Q_log_PHX + neighbor_summation + theta_two * (2*Q_init[i,j] -1)*twenty_images[(img+1)/2,j, i]
    }
  }
  E_Q <- E_Q_log_Q - E_Q_log_PHX
  E_Q_arr[(img+1)/2,iter+1] <- E_Q
  
  
  Q_arr[(img+1)/2,,] <- Q_init
}



E_Q_arr
E_Q_arr[11:20,]

write.table(E_Q_arr[11:12,1:2], file = "energy.csv",row.names = FALSE,col.names = FALSE,sep=",")


show_digit(twenty_images[1,,])







#MAP inference stuff
Q_map_img <- array(0,dim=c(20,28,28))
guess_img <- array(0,dim=c(20,28,28))

for(img in 1:20)
{
  for(r in 1:28)
  {
    for(c in 1:28)
    {
      if (Q_arr[img,r, c] > (1 - Q_arr[img,r, c]))
      {
        Q_map_img[img,r,c] <- 1
        guess_img[img,r,c] <- 1
      }
      else
      {
        Q_map_img[img,r,c] <- 0
        guess_img[img,r,c] <- -1
      }
    }
  }
}

show_digit(twenty_images_original[1,,])
show_digit(guess_img[7,,])
guess_img[1,,]

sample_denoised <- t(array(read.csv('SampleDenoised.csv',header=FALSE)))
show_digit(matrix(sample_denoised[169:196,1:28],nrow=28,ncol=28))
temp_sol_check <- Q_map_img[1:10,,] - array(sample_denoised[1:280,1:28],dim=c(28,28))
matrix(sample_denoised[1:28,1:28],nrow=28,ncol=28)
class(sample_denoised[1:28,1:28])

temp_sol_check

energy_sol <- t(t(array(read.csv('EnergySamples.csv',header=FALSE))))
energy_check <- E_Q_arr[1:10,] - energy_sol


map_csv_output <- matrix(array(0,dim=c(28,280)),nrow=28)
map_csv_output[,1:28] <- Q_map_img[11,,]
map_csv_output[,29:56] <- Q_map_img[12,,]
map_csv_output[,57:84] <- Q_map_img[13,,]
map_csv_output[,85:112] <- Q_map_img[14,,]
map_csv_output[,113:140] <- Q_map_img[15,,]
map_csv_output[,141:168] <- Q_map_img[16,,]
map_csv_output[,169:196] <- Q_map_img[17,,]
map_csv_output[,197:224] <- Q_map_img[18,,]
map_csv_output[,225:252] <- Q_map_img[19,,]
map_csv_output[,253:280] <- Q_map_img[20,,]


write.table(map_csv_output, file = "denoised_2.csv",row.names=FALSE,col.names=FALSE,sep=",")

#ROC stuff

roc_arr <- array(0,dim=c(6,2))
#FPR in first column and TPR in second
#c = { 5, .6 ,.4, .35, .3, .1}

TPR <- array(0,dim=c(6,1))
FPR <- array(0,dim=c(6,1))

new_theta <- 0
new_thetas <- c(5,.6,.4,.35,.3,.1)
for(t in 1:length(new_thetas))
{
  new_theta <- new_thetas[t]
  true_positive <- 0
  false_positive <- 0
  total_positive <- 1
  for(img in 1:20)
  {
    for(r in 1:28)
    {
      for(c in 1:28)
      {
        if(twenty_images_original[img,r,c] == guess_img[img,r,c] && guess_img[img,r,c] == 1)
          true_positive <- true_positive + 1
        else if(twenty_images_original[img,r,c] != guess_img[img,r,c] && guess_img[img,r,c] == 1)
          false_positive <- false_positive + 1
        if(guess_img[img,r,c] == 1)
          total_positive <- total_positive +1
        
      }
    }
    
  }
  TPR[t] <- true_positive/total_positive
  FPR[t] <- false_positive/total_positive
}


